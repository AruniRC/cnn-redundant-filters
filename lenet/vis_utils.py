import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import *



def imshow(img):
    '''
        function to show an image
    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



def setup_cifar_data(batchSize):
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    # The data is saved under './data/cifar-10-batches-py'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                              shuffle=True, num_workers=2) # random shuffling for train dataloader

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  

    return trainloader, testloader, classes               


def vis_kernels(tensor, num_cols=6):
    '''
        This function visualizes the last layer filters of a CNN
        and returns a handle to the figure.
        `tensor` must be a Kx3xWxH filter. 

        Code modified from the original:
        https://discuss.pytorch.org/t/understanding-deep-network-visualize-weights/2060/8
    '''
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[1]==3:
        raise Exception("second dim needs to be 3 to plot: [Kx3xWxH] filters")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        im = np.transpose(tensor[i], (1, 2, 0)) # 3xWxH --> WxHx3
        # rescale pixel values
        low, high = np.min(im), np.max(im)
        im1 = 255.0 * (im - low) / (high - low)

        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(im1.astype('uint8'))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    return fig


def plot_epoch_perf(epoch_accu_train, epoch_accu_val):
    '''
        Return a handle to a figure showing training 
        and validation accuracy per epoch.
    '''
    fig = plt.figure()
    ax = fig.gca()
    epoch_list = range(epoch_accu_train.size)
    ax.plot(epoch_list, epoch_accu_train,label='train')
    ax.plot(epoch_list, epoch_accu_val,label='val')
    ax.grid(True)
    ax.legend()
    ax.set_title('Accuracy')
    ax.set_xlabel(xlabel='epochs')
    return fig


def plot_loss_accu(loss_train, accu_train, t):
    '''
        Return a handle to a figure showing training 
        accuracy and loss. The input `t` is a vector
        of same size as `loss_train` and `accu_train` that
        defines the X-axis points (usually iteration number).
    '''
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(t,loss_train,label='loss') 
    axarr[0].set_title('Loss')
    axarr[0].grid(True)
    axarr[1].plot(t,accu_train,label='accu')
    axarr[1].set_title('Accuracy')
    axarr[1].grid(True)
    return f


def accuracy_on_dataset(net, dataloader, useGpu):
    '''
        Evaluate a network on a given dataset.
    '''
    correct = 0
    total = 0
    useGpu = useGpu & torch.cuda.is_available();
    
    if useGpu:
        net.cuda();

    for data in dataloader:
        images, labels = data
        if useGpu:
            outputs = net(Variable(images.cuda()))
        else:
            outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    return (100 * correct / total)


def predict(net, images):
    '''
        Use `net` to predict the labels of given images.
    '''
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    return predicted


def print_cifar_per_class_accuracy(net, classes, testloader, useGpu):
    '''
        Per-class accuracy on CIFAR (10 class) dataset

    '''
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    useGpu = useGpu & torch.cuda.is_available();
    for data in testloader:
        images, labels = data
        if useGpu:
            outputs = net(Variable(images.cuda()))
        else:
            outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted.cpu() == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def get_layer_cosine_similarity(net, param_name):
    '''
        Returns the matrix of cosine similarity between all filters 
        in a specified layer of a network.
        This works for both conv-nets and fully-connected MLPs.

        Inputs
        ------
        net         - Pytorch network model
        param_name  - string parameter name, e.g. 'conv1.weight'
                      Note: this cannot handle bias tensors.

        Output
        ------
        wSimilMat   - Matrix of all pairwise cosine similarities
                      returned as a Numpy array.
        w           - The 4-D tensor of network weights.

    '''
    params = net.state_dict()
    w = params[param_name]
    _, wSimilMat = weightCovar(w)
    return wSimilMat.numpy(), w.numpy()



def weightCovar(w, do_abs=False):
    '''
       Compute the cosine-similarity between filters of a conv layer.

       Inputs
       ------
        w       - 4-D tensor of (numFilters,inputChannels,filterW,filterH).
        do_abs  - Optional boolean flag to take the absval of covar matrix.

       Outputs
       -------
        off_diag_energy,w_cov   - The pairwise cosine-similarity matrix and 
                                  the energy of its off-diagonal values. 

       Example
       -------
        conv1Params = list(net_wide.conv1.parameters())
        w = conv1Params[0]
        wsum = weightCovar(w)
        print wsum
        
    '''
    num_filters = w.size()
    num_filters = num_filters[0]
    w_reshaped = w.view(num_filters, -1)
    w_normalized = torch.nn.functional.normalize(w_reshaped,2)
    w_cov = torch.mm(w_normalized, torch.t(w_normalized))

    if do_abs:
        w_cov = w_cov.abs()

    off_diag_energy = torch.sum( w_cov - torch.diag(w_cov.diag()) ) / (num_filters**2 - num_filters)
    # mean() instead of sum() to prevent very large numbers
    return off_diag_energy, w_cov



def flatten(img):
    shape = img.size()
    b = img.view(-1)
    return b


def setup_cifar_data_mlp(batchSize):   
    '''
        Setup data loaders for a fully-connected (MLP) network
        
        Images are flattened from 3x32x32 to 3072 vector
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         torchvision.transforms.Lambda(flatten)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                              shuffle=True, num_workers=2) # random shuffling for train dataloader

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def vis_linear_weights(w, num_cols=10):
    '''
        Visualize the first 100 weights from a linear layer 
        in a 10x10 grid and return handle to the figure.
        
        Inputs
        ------
        w      -    Numpy array of dimensions Kx3x32x32.          
        
        Example
        -------
        # Using an MLP network with the first layer called 'fc1'
        fc1Params = list(net.fc1.parameters())
        w = fc1Params[0].data.numpy() # filters
        w.shape
        f = vis_linear_weights(w)
        f.savefig(os.path.join(expDir,'mlp-filter-01.png'), bbox_inches='tight')
        
    '''
    plt.rcParams['figure.figsize'] = (10,10)

    # f = plt.figure()
    subplotCounter = 1
    sz = w.shape

    num_kernels = sz[0]
    num_rows = 1+ num_kernels // num_cols
    
    f = plt.figure(figsize=(num_cols,num_rows))

    for j in range(sz[0]):

        w1 = w[j].reshape(3, 32, 32)
        w1 = np.transpose(w1, (1,2,0))

        # rescale pixel values
        low, high = np.min(w1), np.max(w1)
        im1 = 255.0 * (w1 - low) / (high - low)

        ax = f.add_subplot(num_rows, num_cols, subplotCounter)
        ax.imshow(im1.astype('uint8'))
        subplotCounter += 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    plt.rcParams.update(plt.rcParamsDefault) # reset figure size to defaults
    return f


def vis_filter_conn_comp(outDir, cc_list, layer_curr_weights):
    '''
        Save all the filter connected-component filters as images under `outDir`
    '''
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    for i in range(len(cc_list)):
        f = vis_linear_weights(layer_curr_weights.numpy()[cc_list[i]])
        f.savefig(os.path.join(outDir, ('%d.png' % i)), \
                 bbox_inches='tight')
        plt.close(f)
    plt.rcParams.update(plt.rcParamsDefault)


def vis_closest_filter_pairs(w, similMat, MAX_PAIRS=20):
    '''
        Plots the closest pairs of network filters.
        
        Inputs
        ------
        w         -  Weight matrix (KxD) of K filters of dim D.
        similMat  -  KxK similarity matrix of K filters.
                     Usually obtained by 
                        similMat, w = get_layer_cosine_similarity(net, 'fc1.weight')
        
        MAX_PAIRS -  Optional argument. Default: 20.
        
        Output
        ------
        f         -  Returns handle to the plotted figure.
        pairList  -  List of lists (similarity, rowId, colId)
        
        
        Example
        -------
        import matplotlib.pyplot as plt
        from vis_utils import *

        similMat, w = get_layer_cosine_similarity(net, 'fc1.weight')
        MAX_PAIRS = 20
        outDir = 'data/viz' # assuming an existing folder
        
        f = vis_closest_filter_pairs(similMat, MAX_PAIRS)       
        
        plt.tight_layout()
        f.savefig(os.path.join(outDir,'nearest-filter-pairs.png'), \
                    bbox_inches='tight')
        plt.close(f) # always close the figure after saving to file
    
    '''
    
    # get indices and mask array of lower-triangular matrix
    sz = similMat.shape
    offdiagIndices = np.tril_indices(sz[0], k=-1) # below diagonal
    offdiagVal = similMat[offdiagIndices] 
    maskTril = np.zeros(sz, dtype=bool) # boolean array same shape as similMat
    maskTril[offdiagIndices] = True

    # get the 95-th percentile of cosine similarities
    similThresh = np.percentile(offdiagVal, 95)
    maskLargeSimil = np.greater(similMat, similThresh)
    rowId, colId = np.nonzero(maskLargeSimil & maskTril)

    # sort these in order of similarity
    similValues = similMat[rowId, colId]
    pairList = zip(similValues.tolist(), rowId.tolist(), colId.tolist() )
    pairList.sort()
    pairList.reverse() # descending

    K = np.min([len(pairList), MAX_PAIRS]) # number of filter pairs to display

    # figure settings - display closest pairs
    plt.rcParams["figure.figsize"] = (2,K) # (width, height)
    f = plt.figure()
    subplotCounter = 1

    for j in range(K):

        listItem = pairList[j] # (similarity, rowId, colId)

        w1 = w[listItem[1]]
        w1 = w1.reshape(3, 32, 32)
        w1 = np.transpose(w1, (1,2,0))
        low, high = np.min(w1), np.max(w1)
        im1 = 255.0 * (w1 - low) / (high - low)
        ax = f.add_subplot(K,2, subplotCounter)
        ax.imshow(im1.astype('uint8'))
        subplotCounter += 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('Cosine (%d,%d):' % (listItem[1], listItem[2]))

        w2 = w[listItem[2]]
        w2 = w2.reshape(3, 32, 32)
        w2 = np.transpose(w2, (1,2,0))
        # rescale pixel values
        low, high = np.min(w2), np.max(w2)
        im2 = 255.0 * (w2 - low) / (high - low)
        ax = f.add_subplot(K,2, subplotCounter)
        ax.imshow(im2.astype('uint8'))
        subplotCounter += 1
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('%.3f' % listItem[0])
        
    plt.rcParams.update(plt.rcParamsDefault)
    return f, pairList



def weight_pca(net):
    from sklearn.decomposition import PCA
    w1 = net.fc1.weight.cpu()
    w1 = w1.data.numpy()
    pca = PCA(n_components=w1.shape[0])
    w1_t = pca.fit_transform(w1)
    plt.plot(pca.singular_values_)


def get_dataset_loss(net, dataloader, criterion = nn.CrossEntropyLoss()):
    '''
        Calculate the network loss over a dataset.
        Returns a vector containing the loss for every data point.
    '''
    data_loss = []    
    useGpu = next(net.parameters()).is_cuda
    
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        if useGpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        loss = criterion(outputs, labels)  
        data_loss.append(loss.data.cpu())
        
    data_loss = [x.numpy()[0] for x in data_loss]    
    return data_loss


#--------------------------------------------------------------------------------
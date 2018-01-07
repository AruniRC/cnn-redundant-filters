

# pytorch
import torch
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# general
import os
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# codebase
from vis_utils import *
from model_def import *
from model_train import *
from net_reduce import *


# specify training settings
expName = 'cifar-lenet-v1_w1-500'
w1 = 500 # change conv1 dim
w2 = 50
batchSize = 100
useGpu = True
numEpochs = 30 # orig: 30
learningRate = 0.01 # orig: 0.01 
gamma = 0.1  # learning rate decay
lr_decay_step = 10 # decay lr every "x" epochs
momentum = 0.8
weight_decay = 5e-3
modelPath = [] #'data/cifar-lenet-v1_w1-500/net-reduced-dup-0.80.dat' # specify path if resuming training from saved model, else []

# specify network pruning settings
NET_REDUCE = True
SIMIL_THRESH = 0.8
LAYER_CURR = 'conv1'
LAYER_NEXT = 'conv2'


# experiment folder
expDir = os.path.join('./data', expName)

# setup and load CIFAR dataset
trainloader, testloader, classes = setup_cifar_data(batchSize)




# -----------------------------------------------------------------------------
#   Network training
# -----------------------------------------------------------------------------
# create experiment folder
if not os.path.exists(expDir):
    os.makedirs(expDir)
    
# save training config 
# (NOTE: changing values in the JSON will not reflect in training)
cfg = {'useGpu': useGpu, 'numEpochs': numEpochs, \
       'learningRate': learningRate, \
       'batchSize': batchSize, 'momentum': momentum, \
       'weight_decay': weight_decay, \
       'gamma': gamma, 'w1': w1, 'w2': w2}

# cfg = json.load(file(os.path.join(expDir,'train_config.json'), 'r'))
with open(os.path.join(expDir,'train_config.json'), 'w') as config_file :
    json.dump(cfg, config_file, indent=4, separators=(',', ': '), \
                                          sort_keys=True)

# create ConvNet
net = NetWide(conv1_num_filter=w1, conv2_num_filter=w2)
print(net)
if not modelPath:
    pass # start from random init
else:
    # load network weights from provided `modelPath`
    net.load_state_dict(torch.load(modelPath))

# Define a Loss function and optimizer settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learningRate, \
                      momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=gamma)

if not os.path.isfile(os.path.join(expDir, 'net-trained.dat')):
    # Train the network
    train_cifar_net(net, trainloader, testloader, \
                    criterion, (optimizer,lr_scheduler), expDir, \
                    batchSize=batchSize,
                    numEpochs=numEpochs, useGpu=useGpu, \
                    doVisFilter=False, \
                    fixFilterList=[], verboseFrequency=100)

    torch.save(net.state_dict(), os.path.join(expDir,'net-trained.dat'))
else:
    print 'Loading network state dict from file.'
    net.load_state_dict(torch.load(os.path.join(expDir, 'net-trained.dat')))
    print 'Done.'

# Evaluate
if useGpu:
    net.cuda()

train_acc = accuracy_on_dataset(net, trainloader, useGpu)
print('Accuracy of the network on the 50000 training images: %d %%' % (
    train_acc))

test_acc = accuracy_on_dataset(net, testloader, useGpu)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    test_acc))

res = {'train_acc': train_acc, 'test_acc': test_acc}

# cfg = json.load(file(os.path.join(expDir,'model_acc.json'), 'r'))
with open(os.path.join(expDir,'model_acc.json'), 'w') as res_file :
    json.dump(res, res_file, indent=4, separators=(',', ': '), \
                                          sort_keys=True)

# Visualizations
net.cpu()
similMat, w = get_layer_cosine_similarity(net, 'conv1.weight') # returned as numpy arrays
np.save(os.path.join(expDir,'simil-mat'), similMat)
np.save(os.path.join(expDir,'w-mat'), w)
smat = np.load(os.path.join(expDir,'simil-mat.npy'))
w_mat = np.load(os.path.join(expDir,'w-mat.npy'))

f = plt.figure()
plt.imshow(similMat)
plt.colorbar()
f.savefig(os.path.join(expDir,'lenet-simil-mat.png'), 
                        bbox_inches='tight')





# -----------------------------------------------------------------------------
#   Reduce redundant filters
# -----------------------------------------------------------------------------
if not NET_REDUCE:
    print 'No reduction.'
else:
    # get original data loss
    net.cuda()
    orig_data_loss = np.asarray(get_dataset_loss(net, testloader))

    # Get an adjancency matrix by thresholding the similarity matrix
    net.cpu()
    similMat, _ = get_layer_cosine_similarity(net, LAYER_CURR+'.weight')
    sz = similMat.shape
    adj_mat = np.greater(similMat, SIMIL_THRESH)

    # Find connected components in the graph induced by the adjacency matrix
    cc_list, n_comps = get_adjmat_conn_comp(adj_mat)
    print 'Number of connected comps in graph: %d ' % n_comps


    # Distribution of filter-cc sizes
    cc_sizes = [len(x) for x in cc_list]
    f = plt.figure()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.hist(cc_sizes, bins=range(np.max(cc_sizes)+1), align='left', log=True);
    # plt.xticks(range(np.max(cc_sizes)));
    plt.xlabel('Filter group sizes'); plt.ylabel('log frequency')
    f.savefig(os.path.join(expDir, 'filter-group-size-hist_%.2f.png' % SIMIL_THRESH), \
            bbox_inches='tight')

    # scale filters
    net.cpu()
    scale_net_params(net, LAYER_CURR, LAYER_NEXT)

    # reduction
    reduce_similar_filters(net, LAYER_CURR, LAYER_NEXT, SIMIL_THRESH)

    # Evaluate reduced network's accuracy
    net.eval()
    abl_accu = accuracy_on_dataset(net, testloader, True)
    print 'Reduced network accuracy: %.2f %%' % abl_accu
    torch.save(net.state_dict(), \
             os.path.join(expDir,'net-reduced-dup-%.2f.dat' % SIMIL_THRESH))
    res_dup = {'num_filters': n_comps,'orig_accu': test_acc, 'reduced_accu': abl_accu}
    # cfg = json.load(file(os.path.join(expDir,'model_acc.json'), 'r'))
    with open(os.path.join(expDir,'model_acc_dup-%.2f.json' % SIMIL_THRESH), 'w') as res_file :
        json.dump(res_dup, res_file, indent=4, separators=(',', ': '), \
                                            sort_keys=True)


    # Histogram of network output difference
    net.cuda()
    net.eval()
    new_data_loss = get_dataset_loss(net, testloader)
    new_data_loss = np.asarray(new_data_loss)
    net_diff = np.absolute(new_data_loss - orig_data_loss)
    net_diff_percent = np.divide(net_diff, orig_data_loss) * 100
    f = plt.figure()
    plt.hist(net_diff_percent, normed=True);
    ax = plt.gca()
    plt.xlabel('output change (%)')
    plt.title('Histogram of network output change (mean = %.2f%%)' % np.mean(net_diff_percent))
    # plt.yticks([])
    f.savefig(os.path.join(expDir,'loss-delta-reduced-dup-%.2f.png' % SIMIL_THRESH))


    # -----------------------------------------------------------------------------
    #   Baseline: L1-norm based pruning
    # -----------------------------------------------------------------------------
    # baseline: keep same number of filters as the duplicate method
    NUM_KEEP = n_comps 

    # re-load the original network
    net = NetWide(conv1_num_filter=w1, conv2_num_filter=w2)
    net.load_state_dict(torch.load(os.path.join(expDir, 'net-trained.dat')))

    net.cuda()
    orig_data_loss = get_dataset_loss(net, testloader)
    print 'Original network data loss: %.4f' % np.mean(orig_data_loss)

    reduce_low_norm_filters(net, LAYER_CURR, LAYER_NEXT, NUM_KEEP)

    # Evaluate baseline network's accuracy
    net.cuda()
    net.eval()
    baseline_accu = accuracy_on_dataset(net, testloader, True)
    print 'Reduced network accuracy: %.2f %%' % baseline_accu
    torch.save(net.state_dict(), \
             os.path.join(expDir,'net-reduced-norm-%.2f.dat' % SIMIL_THRESH))
    
    res_norm = {'num_filters': n_comps,'orig_accu': test_acc, 'reduced_accu': baseline_accu}
    with open(os.path.join(expDir,'model_acc_norm-%.2f.json' % SIMIL_THRESH), 'w') as res_file :
        json.dump(res_norm, res_file, indent=4, separators=(',', ': '), \
                                            sort_keys=True)

    # Histogram of network output difference
    net.cuda()
    net.eval()
    new_data_loss = get_dataset_loss(net, testloader)
    new_data_loss = np.asarray(new_data_loss)
    net_diff = np.absolute(new_data_loss - orig_data_loss)
    net_diff_percent = np.divide(net_diff, orig_data_loss) * 100

    f = plt.figure()
    plt.hist(net_diff_percent, normed=True);
    ax = plt.gca()
    plt.xlabel('output change (%)')
    plt.title('Histogram of network output change (mean = %.2f%%)' % np.mean(net_diff_percent))
    # plt.yticks([])
    f.savefig(os.path.join(expDir,'loss-delta-reduced-norm-%.2f.png' % SIMIL_THRESH))



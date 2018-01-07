import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import glob
import sys
from tqdm import tqdm
from vis_utils import *
from model_def import *
from IPython.core.debugger import set_trace


def get_latest_checkpoint_epoch(checkpoint_dir):
    '''
        Return the latest saved network's epoch as an integer.
        If there are no valid checkpoints saved, then return -1.
    '''
    a = glob.glob(os.path.join(checkpoint_dir, 'net*.dat'))
    if not a:
        return -1

    # extract the epoch number from the checkpoint filename
    b = [x.split('-')[-1] for x in a]
    c = [int(x.split('.')[0]) for x in b] 

    return np.max(c)


def adjust_learning_rate(epoch, scheduler):
    '''
        Calls step() on PyTorch scheduler object `epoch+1` times.
        This restores the proper learning rate when resuming training.

        Inputs
        ------
        epoch       -   Integer number denoting the last checkpointed epoch.
        scheduler   -   PyTorch learning rate Scheduler object.

    '''
    for i in range(epoch+1):
        scheduler.step()


def load_from_checkpoint(net, optimizer, expDir, last_checkpoint_epoch):
    print('Resuming training from saved checkpoint.')

    # load net and optimizer to last saved states
    checkpoint_net_path = os.path.join(expDir, 'checkpoints', 
                      'net-epoch-%03d.dat' % last_checkpoint_epoch)
    net.load_state_dict(torch.load(checkpoint_net_path))

    checkpoint_optim_path = os.path.join(expDir, 'checkpoints', 
                      'optim-epoch-%03d.dat' % last_checkpoint_epoch)
    if os.path.exists(checkpoint_optim_path):
        # if no optim checkpointed, use optim settings from input args
        optimizer.load_state_dict(torch.load(checkpoint_optim_path))
    else:
        print 'No optim checkpoint found. Using optim settings from input args.'

    return net, optimizer 

def load_running_average(expDir, last_checkpoint_epoch, isVal=False):
    epoch_accu_train = np.loadtxt(os.path.join(expDir, 'net-accu-train.log'))
    epoch_accu_val = []
    if isVal:
        epoch_accu_val = np.loadtxt(os.path.join(expDir, 'net-accu-val.log'))
    # truncate at last saved epoch
    #   - case: user traines for 10 epochs, deletes last 5 nets and re-starts. 
    #           Logs will show 10 values and must be truncated at 5.
    if epoch_accu_train.size > 1:
        epoch_accu_train = epoch_accu_train[:last_checkpoint_epoch]
    if isVal:
        if epoch_accu_val.size > 1:
            epoch_accu_val = epoch_accu_val[:last_checkpoint_epoch]
        assert epoch_accu_val.size == epoch_accu_train.size
    return epoch_accu_train, epoch_accu_val


def save_batch_stats(expDir, epoch, i, running_loss, running_accu, verboseFrequency, batchSize, optimizer, t, iter_count, loss_train, accu_train, doTrainPlot):

    console_str = 'epoch: %3d batch: %5d loss: %.3f accu: %.3f lr: %f' % \
                   ( epoch, i + 1, running_loss / verboseFrequency, 
                     running_accu/(batchSize*verboseFrequency), 
                     optimizer.param_groups[0]['lr'])
    print(console_str) # print to console

    # print to log file
    with open(os.path.join(expDir,'net-train-console.log'), 'a') \
                                                        as logfile:
        logfile.write(console_str + '\n')
    
    t = np.append(t, iter_count);
    loss_train = np.append(loss_train, 
                        running_loss / verboseFrequency)
    accu_train = np.append(accu_train, 
                        running_accu / (batchSize*verboseFrequency))

    # plotting mini-batch training loss and accuracy
    if doTrainPlot:
        f = plot_loss_accu(loss_train, accu_train, t)
        f.savefig(os.path.join(expDir,'cifar_net_train.png'), 
                                        bbox_inches='tight')
        plt.close(f)

    return t, loss_train, accu_train 
                    


def update_freeze_weights(net, lr, weight_decay, frozen_layer_name=[], freeze_filters = [], new_filter_lr_mult=1):
    ''' 
        Update weights with explicit SGD (no optim) with first layer frozen.
        Loops through all the named modules in a network and updates them 
        using gradient descent. This is useful when freezing some part of a 
        layer -- if Variable.data is modified manually, the optimizer of PyTorch 
        breaks.


        TODO - currently only supports nn.Linear layers (no ConvNets)

        Inputs
        ------
        net                 -   neural net model
        lr                  -   learning rate for the whole network
        weight_decay        -   weight decay for the network
        frozen_layer_name   -   module name to be frozen, e.g. 'fc1'
        freeze_filters      -   subset of filters in a layer to be frozen
        new_filter_lr_mult  -   if a new filter has been added to the layer, 
                                then this allows to specify a higher learning 
                                rate multiplier for it.
    '''
    is_gpu = False
    if next(net.parameters()).is_cuda:
        is_gpu = True

    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            if name == frozen_layer_name:
                # binary mask over the gradient
                if is_gpu:
                    g_weight_mask = \
                        torch.cuda.FloatTensor(m.weight.grad.size()).fill_(1.0)
                else:
                    g_weight_mask = \
                        torch.FloatTensor(m.weight.grad.size()).fill_(1.0)
                g_weight_mask[freeze_filters,:] = 0

                m.weight.data.sub_(m.weight.grad.data * g_weight_mask * lr * new_filter_lr_mult)
                m.weight.data.sub_(m.weight.grad.data * g_weight_mask * weight_decay)
                m.bias.data.sub_(m.bias.grad.data * lr * new_filter_lr_mult)
            else:
                m.weight.data.sub_(m.weight.grad.data * lr)
                m.weight.data.sub_(m.weight.grad.data * weight_decay)
                m.bias.data.sub_(m.bias.grad.data * lr)



def update_perturb_weights(net, lr, weight_decay, layer_name=[], perturbed_filters = []):
    ''' 
        Inputs
        ------
        net                 -   neural net model
        lr                  -   learning rate for the whole network
        weight_decay        -   weight decay for the network
        layer_name          -   module name that is perturbed, e.g. 'fc1'
        perturbed_filters   -   [perturbed_dup_filter_ID, unperturbed_dup_filter_ID]
    '''
    is_gpu = False
    if next(net.parameters()).is_cuda:
        is_gpu = True

    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            if name == layer_name:
                # binary mask over the gradient
                if is_gpu:
                    g_weight_mask = \
                        torch.cuda.FloatTensor(m.weight.grad.size()).fill_(0.0)
                else:
                    g_weight_mask = \
                        torch.FloatTensor(m.weight.grad.size()).fill_(1.0)
                g_weight_mask[perturbed_filters,:] = 1

                m.weight.data.sub_(m.weight.grad.data * g_weight_mask * lr)
                m.weight.data.sub_(m.weight.grad.data * g_weight_mask * weight_decay)
                m.bias.data.sub_(m.bias.grad.data * lr)



def train_cifar_net(net, trainloader, valloader, criterion, optimizer, expDir, 
                    batchSize=32,  numEpochs=2, useGpu=False, fixFilterList=[], 
                    verboseFrequency=500, doTrainPlot=True, doVisFilter=True, 
                    numNewFilter=0, newWeightLrMult=1, doVisSimilMat=False, 
                    perturbed_filter=[]):
    '''
        Inputs
        ------
        optimizer       - A torch.optim.SGD object. Can be any other solver too.
                          A second option allows this to a 2-tuple of 
                          (optimizer, scheduler), providing a way to send in a 
                          learning rate scheduler without changing the function 
                          signature.

        fixFilterList   - A list of indices of which filters are *not* to be 
                          updated during training. Default: [].
        numNewFilter    - Number of newly added filters in 1st layer. Default:0.
        newWeightLrMult - Learning rate multiplier for newly-added (unfrozen) 
                          weights. Default: 1.


        Notes
        -----
        1. Checkpoints:
        This saves checkpoints by default under <experimentDir>/checkpoints.
        If training is interrupted and train_cifar_net() is called again with 
        the same expDir, then it reads in the saved state_dicts of the network 
        and the optimizer. 

        2. Changing the learning rate:
        There are two ways of changing the learning rate over epochs. The first 
        way is to use the lr_scheduler from PyTorch. If we manually need to 
        change the learning rate from, say 0.01 to 0.001 at 10th epoch, we have 
        to interrupt training at 9th epoch, delete the optim-checkpoint of epoch 
        9 (but keep the net checkpoint), set learning rate of optimizer to 0.001 
        and then call the train_cifar_net() function once more.

    '''
    
    # --------------------------------------------------------------------------
    #  Initial setup
    # --------------------------------------------------------------------------
    
    # plotting training stats
    loss_train = []
    accu_train = []
    iter_count = 0
    t = []
    epoch_accu_train = []
    epoch_accu_val = []

    # make folders to save filter visualizations during training
    if not os.path.exists(os.path.join(expDir, 'conv-filters')):
        os.makedirs(os.path.join(expDir, 'conv-filters'))  
        
    if not os.path.exists(os.path.join(expDir, 'conv-filters-pairs')):
        os.makedirs(os.path.join(expDir, 'conv-filters-pairs'))

    # make training checkpoint folder
    if not os.path.exists(os.path.join(expDir, 'checkpoints')):
        os.makedirs(os.path.join(expDir, 'checkpoints'))
        last_checkpoint_epoch = -1
    else:
        # check for latest saved checkpoint
        last_checkpoint_epoch = get_latest_checkpoint_epoch(
                                    os.path.join(expDir, 'checkpoints'))
    
    # shift net to GPU
    if useGpu:
        net.cuda();

    # hack: wrap both optimizer and learning rate scheduler into a single 
    # 2-tuple: (optimizer, scheduler)
    if type(optimizer) is tuple:
        scheduler = optimizer[1]
        optimizer = optimizer[0]

    opt_state_dict = optimizer.state_dict() # optim setting from input args
        
    # --------------------------------------------------------------------------
    #      Training loop
    # --------------------------------------------------------------------------
    for epoch in tqdm(range(numEpochs)):

        # resume training from last saved checkpoint
        if last_checkpoint_epoch >= 0:         # training is resuming, not from scratch
            if epoch <= last_checkpoint_epoch:  # fast forward to (latest_epoch + 1)
                continue
            else:
                # load checkpoint network and optimizer
                net, optimizer = \
                load_from_checkpoint(net, optimizer, expDir, last_checkpoint_epoch)

                if 'scheduler' in locals():
                    adjust_learning_rate(last_checkpoint_epoch, scheduler)

                # update the running averages of training stats
                if 'valloader' in locals():
                    isVal = True
                else:
                    isVal = False
                epoch_accu_train, epoch_accu_val = \
                    load_running_average(expDir, last_checkpoint_epoch, isVal)

                # normal training from next epoch onwards (no resuming from checkpoint)
                last_checkpoint_epoch = -1

        running_loss = 0.0
        running_accu = 0.0

        # if a learning rate scheduler exists then change lr every epoch
        if 'scheduler' in locals():
            scheduler.step()
        
        for i, data in enumerate(trainloader, 0):
            
            # ------------------------------------------------------------------
            #      Mini-batch loading and optimization
            # ------------------------------------------------------------------
            
            # set network to train mode
            #  net.train(True)
            
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if useGpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                net.cuda();
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                net.cpu();

            # zero the parameter gradients
            optimizer.zero_grad()

            for param in net.parameters():
                param.requires_grad = True

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)     
            loss.backward()


            # ------------------------------------------------------------------
            # Opt 1: specify a different learning rate for newly added params
            if numNewFilter > 0:
                frozen_layer_name = 'fc1'
                freeze_filters =  fixFilterList
                new_filter_lr_mult = newWeightLrMult    

            # ------------------------------------------------------------------
            # Opt 2: freeze a subset of the first-layer filters
            # set specified filters in first-layer to have no gradient updates
            if fixFilterList or perturbed_filter:
                # manual weight update
                learning_rate = optimizer.param_groups[0]['lr']
                weight_decay = optimizer.param_groups[0]['weight_decay']

                if fixFilterList:
                    frozen_layer_name = 'fc1'
                    freeze_filters =  fixFilterList
                    update_freeze_weights(net, learning_rate, weight_decay,\
                                          frozen_layer_name,\
                                          freeze_filters,\
                                          new_filter_lr_mult)
                else:
                    perturb_layer = 'fc1' # fixed for now
                    update_perturb_weights(net, learning_rate, weight_decay, \
                                   perturb_layer, perturbed_filter)

            else:
                frozen_layer_name = []
                freeze_filters = []
                optimizer.step()  # default optimizer

            
            # ------------------------------------------------------------------
            #      Network mini-batch training statistics
            # ------------------------------------------------------------------
            
            # set network to eval
            # net.eval()
            
            # calculate training accuracy
            preds = predict(net, inputs.data)
            accu = (preds == labels.data).sum()  # `.data` since labels is wrapped in Variable
            iter_count = iter_count + 1
            running_loss += loss.data[0]
            running_accu += accu
            
            # display stats every `verboseFrequency` iterations
            if i % verboseFrequency == (verboseFrequency-1):
                t, loss_train, accu_train = save_batch_stats(expDir, epoch, i, \
                     running_loss, running_accu, verboseFrequency, \
                     batchSize, optimizer, t, iter_count, loss_train, accu_train, \
                     doTrainPlot)


                # --------------------------------------------------------------
                #      First layer weight visualizations
                # --------------------------------------------------------------
                if doVisFilter:
                    
                    # visualize filters for conv-nets (first layer is Conv2d)
                    if not isinstance(net, MLP):
                        # plot and save current filters
                        w = net.conv1.weight.data
                        fig = vis_kernels(w.cpu().numpy())
                        fig.savefig(os.path.join(expDir, 'conv-filters', 
                                str(iter_count) + '.png'), bbox_inches='tight')
                        
                        # update filters on same image
                        fig.savefig(os.path.join(expDir,'conv-filters.png'), 
                                    bbox_inches='tight') 
                        plt.close(fig)

                    # visualize weights for MLP (first layer is Linear)    
                    if isinstance(net, MLP):
                        fc1Params = list(net.fc1.parameters())
                        w = fc1Params[0].cpu().data.numpy()
                        f = vis_linear_weights(w, num_cols=50)
                        f.savefig(os.path.join(expDir, 'conv-filters', 
                                str(iter_count) + '.png'), bbox_inches='tight')
                        plt.close(f)

                # save filter similarity matrices over iterations
                if doVisSimilMat:
                    if isinstance(net, MLP):
                        LAYER_CURR = 'fc1'
                        net.cpu()
                        similMat, _ = get_layer_cosine_similarity(net, \
                                                        LAYER_CURR+'.weight')
                        fig = plt.figure()
                        plt.imshow(similMat)
                        plt.colorbar()
                        plt.xticks([])
                        plt.yticks([])
                        outDir = os.path.join(expDir, 'simil-mats')
                        if not os.path.exists(outDir):
                            os.makedirs(outDir)

                        fig.savefig(os.path.join(outDir,'mlp-batch_%d.pdf' % iter_count), \
                                  bbox_inches='tight')
                        fig.savefig(os.path.join(outDir,'mlp-batch_%d.png' % iter_count), \
                                  bbox_inches='tight')
                    
                running_loss = 0.0
                running_accu = 0.0


        # ----------------------------------------------------------------------
        #      Network epoch training statistics
        # ----------------------------------------------------------------------
        # net.eval()
        if useGpu:
            net.cuda();
        epoch_accu_train = np.append(epoch_accu_train, 
                            accuracy_on_dataset(net, trainloader, useGpu))
        if valloader:
            epoch_accu_val = np.append(epoch_accu_val, 
                            accuracy_on_dataset(net, valloader, useGpu))
            assert epoch_accu_val.size == epoch_accu_train.size

        # save training checkpoints at each epoch:
        #   save the network and optimizer state under 
        #       <expDir>/checkpoints/
        torch.save(net.state_dict(), os.path.join(expDir, 'checkpoints', 
                                     'net-epoch-%03d.dat' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(expDir, 'checkpoints', 
                                     'optim-epoch-%03d.dat' % epoch))

        # save training logs
        np.savetxt(os.path.join(expDir, 'net-accu-train.log'), 
                    epoch_accu_train)
        if valloader:
            np.savetxt(os.path.join(expDir, 'net-accu-val.log'), 
                    epoch_accu_val)
        
        # show epoch-summary plots at end of each epoch
        if doTrainPlot:
            fig = plt.figure()
            ax = fig.gca()
            epoch_list = range(epoch_accu_train.size)
            ax.plot(epoch_list, epoch_accu_train,label='train')
            if valloader:
                ax.plot(epoch_list, epoch_accu_val,label='val')
            ax.grid(True)
            ax.legend()
            ax.set_title('Accuracy')
            ax.set_xlabel(xlabel='epochs')
            fig.savefig(os.path.join(expDir,'net-train-epoch.png'), 
                        bbox_inches='tight')
            plt.close(fig)


    # Training completed: shift net back to CPU and set to eval mode
    net.cpu()
    # net.eval()
    print('Finished Training')


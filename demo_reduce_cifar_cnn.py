

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
import numpy as np

# codebase
import lenet
import lenet.vis_utils
import lenet.model_def
import lenet.net_reduce


modelPath = [] #'data/cifar-lenet-v1_w1-500/net-reduced-dup-0.80.dat' # specify path if resuming training from saved model, else []


# TODO - make into command-line args

# specify network pruning settings
NET_REDUCE = 'norm'
NUM_KEEP = 250
SIMIL_THRESH = 0.8
LAYER_CURR = 'conv1'
LAYER_NEXT = 'conv2'

w1 = 500
w2 = 50
modelPath = 'data/cifar-lenet-v1_w1-500/net-trained.dat'
batchSize = 100
expName = 'reduce_cifar-cnn_'+'w1-'+str(w1)
expDir = os.path.join('./data', expName)
useGpu = True



# setup and load CIFAR dataset
trainloader, testloader, classes = lenet.vis_utils.setup_cifar_data(batchSize)

# experiment outputs
if not os.path.exists(expDir):
    os.makedirs(expDir)

# create ConvNet
net = lenet.model_def.NetWide(conv1_num_filter=w1, conv2_num_filter=w2)
print(net)
if not modelPath:
    pass # start from random init
else:
    # load network weights from provided `modelPath`
    net.load_state_dict(torch.load(modelPath))

# Evaluate
if useGpu:
    net.cuda()

net.eval()
test_acc = lenet.vis_utils.accuracy_on_dataset(net, testloader, useGpu)



# ---------------------------------------------------------------------------------
#   Reduce redundant filters
# ---------------------------------------------------------------------------------
if NET_REDUCE == 'duplicate':

    # -----------------------------------------------------------------------------
    #   Baseline: duplicate-filter based pruning
    # -----------------------------------------------------------------------------

    # Get an adjancency matrix by thresholding the similarity matrix
    net.cpu()
    similMat, _ = lenet.vis_utils.get_layer_cosine_similarity(net, LAYER_CURR+'.weight')
    sz = similMat.shape
    adj_mat = np.greater(similMat, SIMIL_THRESH)

    # Find connected components in the graph induced by the adjacency matrix
    cc_list, n_comps = lenet.net_reduce.get_adjmat_conn_comp(adj_mat)
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
    lenet.net_reduce.scale_net_params(net, LAYER_CURR, LAYER_NEXT)

    # reduction
    lenet.net_reduce.reduce_similar_filters(net, LAYER_CURR, LAYER_NEXT, SIMIL_THRESH)

    # Evaluate reduced network's accuracy
    net.eval()
    abl_accu = lenet.vis_utils.accuracy_on_dataset(net, testloader, useGpu)
    print 'Reduced network accuracy: %.2f %%' % abl_accu
    torch.save(net.state_dict(), \
             os.path.join(expDir,'net-reduced-dup-%.2f.dat' % SIMIL_THRESH))
    res_dup = {'num_filters': n_comps,'orig_accu': test_acc, 'reduced_accu': abl_accu}
    
    with open(os.path.join(expDir,'model_acc_dup-%.2f.json' % SIMIL_THRESH), 'w') as res_file :
        json.dump(res_dup, res_file, indent=4, separators=(',', ': '), \
                                            sort_keys=True)

elif NET_REDUCE == 'norm':

    # -----------------------------------------------------------------------------
    #   Baseline: L1-norm based pruning
    # -----------------------------------------------------------------------------

    lenet.net_reduce.reduce_low_norm_filters(net, LAYER_CURR, LAYER_NEXT, NUM_KEEP)

    # Evaluate baseline network's accuracy
    net.cuda()
    net.eval()
    baseline_accu = lenet.vis_utils.accuracy_on_dataset(net, testloader, True)
    print 'Reduced network accuracy: %.2f %%' % baseline_accu
    torch.save(net.state_dict(), \
             os.path.join(expDir,'net-reduced-norm-%.2f.dat' % NUM_KEEP))
    
    res_norm = {'num_filters': NUM_KEEP,'orig_accu': test_acc, 'reduced_accu': baseline_accu}
    with open(os.path.join(expDir,'model_acc_norm-%.2f.json' % NUM_KEEP), 'w') as res_file :
        json.dump(res_norm, res_file, indent=4, separators=(',', ': '), \
                                            sort_keys=True)

else:
    raise ValueError('Valid reduction methods: `norm` and `duplicate`.')


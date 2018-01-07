

# pytorch
import torch
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# general
import argparse
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


# ---------------------------------------------------------------------------------
#
# Loads a pre-trained model from location `modelPath`, which can be downloaded 
# from: http://fisher.cs.umass.edu/~arunirc/downloads/cnn-duplicate/cifar-lenet-v1_w1-500/. 
#
# After placing this model under `./data/cifar-lenet-v1_w1-500`, we will specify 
# the type of pruning (duplicate filters or low-norm filters) using `NET_REDUCE`. 
#
# If pruning is 'norm', then `NUM_KEEP` is the top-k of filters to retain, sorted by norm.
# If pruning is 'duplicates', then `SIMIL_THRESH` decides the cosine-similarity value beyond 
# which two filters would be merged. 
#
# The experiment outputs are saved under `expDir` specified below.
#
# ---------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')

parser.add_argument('--NET_REDUCE', default='duplicate', type=str,
                    help='`norm` or `duplicate` (default)')
parser.add_argument('--model_path', default='data/cifar-lenet-v1_w1-500/net-trained.dat', type=str,
                    help='Pre-trained CIFAR-10 LeNet model')

parser.add_argument('--NUM_KEEP', default=250, type=int,
                    help='Filters to keep in norm-based pruning (default 250)')
parser.add_argument('--SIMIL_THRESH', default=0.8, type=float,
                    help='Cosine-similarity threshold in duplicates-based pruning (default 0.8)')
parser.add_argument('--LAYER_CURR', default='conv1', type=str,
                    help='layer name to prune filters from')
parser.add_argument('--LAYER_NEXT', default='conv2', type=str,
                    help='layer succeeding pruned layer')

parser.add_argument('--w1', default=500, type=int,
                    help='conv1 size (default: 500)')
parser.add_argument('--w2', default=50, type=int,
                    help='conv2 size (default: 50)')

parser.set_defaults(augment=True)


def main():

    global args
    args = parser.parse_args()

    batchSize = 100
    expName = 'reduce_cifar-cnn_'+'w1-'+str(args.w1)
    expDir = os.path.join('./data', expName) # TODO - add timestamp?
    useGpu = True    

    # setup and load CIFAR-10 dataset
    trainloader, testloader, classes = lenet.vis_utils.setup_cifar_data(batchSize)

    # experiment outputs
    if not os.path.exists(expDir):
        os.makedirs(expDir)

    # create ConvNet
    net = lenet.model_def.NetWide(conv1_num_filter=args.w1, conv2_num_filter=args.w2)
    print(net)
    if not args.model_path:
        pass # start from random init
    else:
        # load network weights from provided `model_path`
        net.load_state_dict(torch.load(args.model_path))

    # Evaluate
    if useGpu:
        net.cuda()

    net.eval()
    test_acc = lenet.vis_utils.accuracy_on_dataset(net, testloader, useGpu)
    print 'Original network accuracy: %.2f %%' % test_acc


    # ---------------------------------------------------------------------------------
    #   Reduce redundant filters
    # ---------------------------------------------------------------------------------
    if args.NET_REDUCE == 'duplicate':

        # -----------------------------------------------------------------------------
        #   Baseline: duplicate-filter based pruning
        # -----------------------------------------------------------------------------

        # Get an adjancency matrix by thresholding the similarity matrix
        net.cpu()
        similMat, _ = lenet.vis_utils.get_layer_cosine_similarity(
                                    net, args.LAYER_CURR+'.weight')
        sz = similMat.shape
        adj_mat = np.greater(similMat, args.SIMIL_THRESH)

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
        f.savefig(os.path.join(expDir, 'filter-group-size-hist_%.2f.png' % args.SIMIL_THRESH), 
                bbox_inches='tight')

        # scale filters
        net.cpu()
        lenet.net_reduce.scale_net_params(net, args.LAYER_CURR, args.LAYER_NEXT)

        # reduction
        lenet.net_reduce.reduce_similar_filters(net, args.LAYER_CURR, 
                                                args.LAYER_NEXT, args.SIMIL_THRESH)

        # Evaluate reduced network's accuracy
        net.eval()
        abl_accu = lenet.vis_utils.accuracy_on_dataset(net, testloader, useGpu)
        print 'Reduced network accuracy: %.2f %%' % abl_accu
        torch.save(net.state_dict(), \
                 os.path.join(expDir,'net-reduced-dup-%.2f.dat' % args.SIMIL_THRESH))
        res_dup = {'num_filters': n_comps,'orig_accu': test_acc, 'reduced_accu': abl_accu}
        
        with open(os.path.join(expDir,'model_acc_dup-%.2f.json' % args.SIMIL_THRESH), 'w') as res_file :
            json.dump(res_dup, res_file, indent=4, separators=(',', ': '),
                                                sort_keys=True)

    elif NET_REDUCE == 'norm':

        # -----------------------------------------------------------------------------
        #   Baseline: L1-norm based pruning
        # -----------------------------------------------------------------------------

        lenet.net_reduce.reduce_low_norm_filters(
                            net, args.LAYER_CURR, args.LAYER_NEXT, args.NUM_KEEP)

        # Evaluate baseline network's accuracy
        net.cuda()
        net.eval()
        baseline_accu = lenet.vis_utils.accuracy_on_dataset(net, testloader, True)
        print 'Reduced network accuracy: %.2f %%' % baseline_accu
        torch.save(net.state_dict(),
                 os.path.join(expDir,'net-reduced-norm-%.2f.dat' % args.NUM_KEEP))
        
        res_norm = {'num_filters': NUM_KEEP,'orig_accu': test_acc, 'reduced_accu': baseline_accu}
        with open(os.path.join(expDir,'model_acc_norm-%.2f.json' % args.NUM_KEEP), 'w') as res_file :
            json.dump(res_norm, res_file, indent=4, separators=(',', ': '),
                                                sort_keys=True)

    else:
        raise ValueError('Valid reduction methods: `norm` and `duplicate`.')

    print 'Experiment outputs saved under: %s' % expDir


if __name__ == '__main__':
    main()


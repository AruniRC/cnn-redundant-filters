
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.sparse import *
from vis_utils import *
from model_def import *
from model_train import *



def get_param(net, param_name):
    params = net.state_dict()
    w = params[param_name]
    return w


def get_adjmat_conn_comp(adj_mat):
    '''
        Find connected components given an undirected adjacency matrix. 
        Requires the sparse module from SciPy.

    '''
    sz = adj_mat.shape
    n_comps, comp_labels = csgraph.connected_components(adj_mat, directed=False)
    cc_set = np.unique(comp_labels)

    # init a list of lists
    cc_list = []
    for i in range(len(cc_set)):
        cc_list.append([])

    # cc_list[i] stores the indices (j) of the filters in the i-th conn-comp
    for i in range(len(cc_set)):
        for j in range(sz[0]):
            if comp_labels[j] == cc_set[i]:
                cc_list[i].append(j)

    return cc_list, n_comps


def scale_net_params(net, LAYER_CURR, LAYER_NEXT):
    '''
        Scales the network layers specified in the input args.
        Unit-normalizes the current layer's filters.
        Applies the same scaling on the current biases. 
        Applies the inverse scaling on the next layer's filters.

    '''
    do_reshape = False

    # transfer network to CPU if on GPU
    if next(net.parameters()).is_cuda:
        net.cpu()

    # get the parameters
    layer_curr_weights = get_param(net, LAYER_CURR+'.weight')
    layer_curr_biases = get_param(net, LAYER_CURR+'.bias')
    layer_next_weights = get_param(net, LAYER_NEXT+'.weight')

    if layer_curr_weights.dim() != layer_next_weights.dim():
        # a 4-D tensor followed by a 2-D matrix - dim mismatch!
        #    Conv --> reshape --> Linear
        # TODO - raise exception if sizes mismatch
        do_reshape = True


    # Scale filter weights and biases by the inverse of the filter norms
    w_n = [torch.norm(w.view(-1),p=2) \
            for w in layer_curr_weights] # 2-norm of flattened cxhxw kernels
    w_normed = [w.div(w_n[index]) \
            for index, w in enumerate(layer_curr_weights)] # normalize
    w_normed = torch.stack(w_normed, dim=0)
    b_normed = torch.FloatTensor([b/w_n[index] \
                for index, b in enumerate(layer_curr_biases)])

    # Scale the next layer weights by the current layer's filter norms
    layer_next_scaled = []     
    if do_reshape:
        num_filters = layer_curr_weights.size()[0]
        num_rep = layer_next_weights.size()[1] // num_filters
        w_n = np.repeat(w_n, num_rep)


    for i in range(layer_next_weights.size()[1]):
        layer_next_scaled.append( layer_next_weights[:,i] * w_n[i] )
    layer_next_scaled = torch.stack(layer_next_scaled, dim=1)

    # Update the parameters of the network
    param_dict = net.state_dict()
    param_dict[LAYER_CURR+'.weight'] = w_normed
    param_dict[LAYER_CURR+'.bias'] = b_normed
    param_dict[LAYER_NEXT+'.weight'] = layer_next_scaled
    net.load_state_dict(param_dict)


def reduce_similar_filters(net, LAYER_CURR, LAYER_NEXT, SIMIL_THRESH):
    # transfer network to CPU if on GPU
    if next(net.parameters()).is_cuda:
        net.cpu()
    do_reshape = False

    # threshold filter similarity and get conn-comps
    similMat, _ = get_layer_cosine_similarity(net, LAYER_CURR+'.weight')
    adj_mat = np.greater(similMat, SIMIL_THRESH)
    cc_list, n_comps = get_adjmat_conn_comp(adj_mat)
     
    # scale the current layer weights to unit norm, up-scale next layer
    scale_net_params(net, LAYER_CURR, LAYER_NEXT)

    layer_curr_weights = get_param(net, LAYER_CURR+'.weight')
    layer_curr_biases = get_param(net, LAYER_CURR+'.bias')
    layer_next_weights = get_param(net, LAYER_NEXT+'.weight')
    next_sz = layer_next_weights.size()

    if layer_curr_weights.dim() != layer_next_weights.dim():
        # a 4-D tensor followed by a 2-D matrix - dim mismatch!
        #    forward: Conv --> reshape --> Linear
        # TODO - raise exception if langths mismatch
        do_reshape = True

    # Reduce CURRENT layer FILTERS - mean
    layer_curr_filter_groups = [layer_curr_weights[torch.LongTensor(x)] \
                                    for x in cc_list]
    layer_curr_filter_reduced = [x.mean(dim=0) \
                                    for x in layer_curr_filter_groups]
    layer_curr_filter_cat = torch.stack(layer_curr_filter_reduced, dim=0)

    # Reduce CURRENT layer BIASES - mean
    layer_curr_bias_groups = [layer_curr_biases[torch.LongTensor(x)] \
                                for x in cc_list]
    layer_curr_bias_reduced = [x.mean() \
                                for x in layer_curr_bias_groups]
    layer_curr_bias_cat = torch.FloatTensor(layer_curr_bias_reduced)
    
    
    if do_reshape:
        # reshape next-layer (fully-connected) from 2-D to 4-D
        num_filters = layer_curr_weights.size()[0]
        num_rep = layer_next_weights.size()[1] // num_filters
        sp_dim = np.sqrt(num_rep).astype('int')
        layer_next_weights = \
            layer_next_weights.view(next_sz[0], num_filters, sp_dim, sp_dim)

    # Reduce NEXT layer FILTERS - sum
    layer_next_groups = [layer_next_weights[:,x] for x in cc_list]
    layer_next_reduced = [x.sum(dim=1) for x in layer_next_groups]
    layer_next_cat = torch.stack(layer_next_reduced, dim=1)

    if do_reshape:
        # reshape next-layer back to 2-D
        layer_next_cat = layer_next_cat.view(next_sz[0],-1)


    # Update the parameters of the network
    #  - HACK: PyTorch does not allow loading a state_dict when sizes change
    params = net.named_parameters() # generator
    s = set([LAYER_CURR+'.weight', \
             LAYER_CURR+'.bias', \
             LAYER_NEXT+'.weight'])
    param_dict = {y[0]: y[1] for y in filter(lambda x: x[0] in s, params)}

    param_dict[LAYER_CURR+'.weight'].data = layer_curr_filter_cat
    param_dict[LAYER_CURR+'.bias'].data = layer_curr_bias_cat
    param_dict[LAYER_NEXT+'.weight'].data = layer_next_cat

    return n_comps, cc_list, similMat, adj_mat


def reduce_low_norm_filters(net, LAYER_CURR, LAYER_NEXT, NUM_KEEP):

    # transfer network to CPU if on GPU
    if next(net.parameters()).is_cuda:
        net.cpu()
    do_reshape = False

    # get parameters
    layer_curr_weights = get_param(net, LAYER_CURR+'.weight')
    layer_curr_biases = get_param(net, LAYER_CURR+'.bias')
    layer_next_weights = get_param(net, LAYER_NEXT+'.weight')
    next_sz = layer_next_weights.size()

    # sort by L1 norms
    w_norms = [torch.norm(w.view(-1), p=1) for w in layer_curr_weights]
    sort_indices = np.argsort(w_norms)
    w_norms = np.asarray(w_norms)
    pruned_indices = sort_indices[-NUM_KEEP:]

    if layer_curr_weights.dim() != layer_next_weights.dim():
        # a 4-D tensor followed by a 2-D matrix - dim mismatch!
        #    forward: Conv --> reshape --> Linear
        do_reshape = True
        num_filters = layer_curr_weights.size()[0]
        num_rep = layer_next_weights.size()[1] // num_filters
        sp_dim = np.sqrt(num_rep).astype('int')
        layer_next_weights = \
            layer_next_weights.view(next_sz[0], num_filters, sp_dim, sp_dim)

    # current layer filter reduction
    layer_curr_filter_reduced = layer_curr_weights[torch.LongTensor(pruned_indices)]
    layer_curr_bias_reduced = layer_curr_biases[torch.LongTensor(pruned_indices)]

    # next layer filter reduction
    layer_next_reduced = layer_next_weights[:,torch.LongTensor(pruned_indices)]

    if do_reshape:
        # reshape next-layer back to 2-D
        layer_next_reduced = layer_next_reduced.view(next_sz[0],-1)


    # Update the parameters of the network
    params = net.named_parameters() # generator
    s = set([LAYER_CURR+'.weight', \
             LAYER_CURR+'.bias', \
             LAYER_NEXT+'.weight'])
    param_dict = {y[0]: y[1] for y in filter(lambda x: x[0] in s, params)}

    param_dict[LAYER_CURR+'.weight'].data = layer_curr_filter_reduced
    param_dict[LAYER_CURR+'.bias'].data = layer_curr_bias_reduced
    param_dict[LAYER_NEXT+'.weight'].data = layer_next_reduced





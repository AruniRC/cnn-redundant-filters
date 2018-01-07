# Network definition
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # LeNet-5
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class NetWide(nn.Module):
    '''
        Initialise a conv network with an argument to increase the number of 
        conv1 layer filters.self
    '''
    def __init__(self, conv1_num_filter=64, conv2_num_filter=16):

        super(NetWide, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1_num_filter, 5) # (in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_num_filter, conv2_num_filter, 5)
        self.fc1 = nn.Linear(conv2_num_filter * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.c1 = conv1_num_filter
        self.c2 = conv2_num_filter

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.c2 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MLP(nn.Module):
    '''
        Initialise a fully connected multi-layer perceptron (MLP)
    '''
    def __init__(self, input_dim=3072, fc1_dim=100, fc2_dim=100):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 10)


    def _init_weights(self, init_sigma=0.01):
        '''
            Initialization schemes for layer weights
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, init_sigma)
                m.bias.data.zero_()


    def _add_new_weight(self, num_new_weights = 1, init_sigma=0):
        '''
            Add new column(s) to the first layer's weight matrix
        '''

        params = self.parameters()
        p1 = params.next() # layer 1 weights
        p2 = params.next() # layer 1 biases
        p3 = params.next() # layer 2 weights

        # add a new column to layer 1 weights
        sz = p1.data.size()
        if init_sigma == 0:
            new_col = torch.ones(num_new_weights, sz[1])
        else:
            new_col = init_sigma * torch.randn(num_new_weights, sz[1])
        p1.data = torch.cat((p1.data, new_col), 0)

        # layer 1 bias
        p2.data = torch.squeeze(torch.cat((p2.data, torch.ones(num_new_weights)), 0))

        # add a new row to layer 2 weights (zeros)
        sz_p3 = p3.data.size()
        p3.data = torch.cat((p3.data, \
                             torch.zeros(sz_p3[0],num_new_weights)), \
                            1)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

from __future__ import division
import math
import torch
from torch import nn
import torch.nn.functional as F

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """
    def __init__(self, in_channels, out_channels, orders, activation = 'relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0] # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0) #(num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        # if self.activation == 'relu':
        #     x = F.relu(x)
        # elif self.activation == 'selu':
        #     x = F.selu(x)

        return x


class C_GCN(nn.Module):
    """
    Neural network block that applies a chebynet to sampled location.
    """
    def __init__(self, in_channels, out_channels, orders, activation = 'relu'):

        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The order of convolution
        :param num_nodes: Number of nodes in the graph.
        """
        super(C_GCN, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * orders,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.orders = orders
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        list_cheb = list()
        for k in range(self.orders):
            if (k==0):
                list_cheb.append(torch.diag(torch.ones(A_hat.shape[0],)))
            elif (k==1):
                list_cheb.append(A_hat)
            else:
                list_cheb.append(2*torch.matmul(A_hat, list_cheb[k-1])  - list_cheb[k-2])

        features = list()
        for k in range(self.orders):
            features.append(torch.einsum("kk,bkj->bkj", [list_cheb[k], X]))
        features_cat = torch.cat(features, 2)
        t2 = torch.einsum("bkj,jh->bkh", [features_cat, self.Theta1])
        t2 += self.bias
        if self.activation == 'relu':
            t2 = F.relu(t2)
        if self.activation == 'selu':
            t2 = F.selu(t2)
        return t2



class K_GCN(nn.Module):
    """
    Neural network block that applies a graph convolution to to sampled location.
    """
    def __init__(self, in_channels, out_channels, activation = 'selu'):

        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        :relu is not good for K_GCN on Kriging, so we suggest 'selu'
        """
        super(K_GCN, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_hat: The normalized adajacent matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        features = torch.einsum("kk,bkj->bkj", [A_hat, X])
        t2 = torch.einsum("bkj,jh->bkh", [features, self.Theta1])
        t2 += self.bias
        if self.activation == 'relu':
            t2 = F.relu(t2)
        if self.activation == 'selu':
            t2 = F.selu(t2)

        return t2

class GAT(nn.Module):
    """
    Neural network block that applies attention mechanism to sampled locations (only the attention).
    """
    def __init__(self, in_channels, alpha, threshold, concat = True):
        """
        :param in_channels: Number of time step.
        :param alpha: alpha for leaky Relu.
        :param threshold: threshold for graph connection
        :param concat: whether concat features
        :It should be noted that the input layer should use linear activation
        """
        super(GAT, self).__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.concat = concat
        self.in_channels = in_channels
        self.a = nn.Parameter(torch.zeros(size=(2*in_channels, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

          # num of nodes
        h = input
        B = h.size()[0]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, self.in_channels), h.repeat(1, N, 1)], dim=2).view(B, N, N, 2 * self.in_channels)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj.unsqueeze(0).repeat(B, 1, 1) > self.threshold, e, zero_vec) #>threshold for attention connection

        attention = F.softmax(attention, dim=2)

        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class ResBlock(nn.Module):
    def __init__(self, channel, hidden_channel, order=2):
        super().__init__()
        self.channel = channel
        self.hidden_channel = hidden_channel
        self.order = order

        # SE part
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc1 = nn.Linear(self.hidden_channel, self.hidden_channel, bias=True)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(self.hidden_channel, self.hidden_channel, bias=True)
        self.act2 = nn.Sigmoid()
        self.initialize_kernel()

        # GNN part
        self.GNN = D_GCN(self.hidden_channel, self.hidden_channel, self.order)
        self.act3 = nn.SiLU()

    def initialize_kernel(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x, A_q, A_h):
        # Do GNN part
        x = self.GNN(x, A_q, A_h)

        # Do SE part, attention should be paid to hidden_dim
        x = x.permute(0,2,1)

        se = self.avgpool(x)
        se = torch.permute(se, (0,2,1))
        se = self.act1(self.fc1(se))
        se = self.act2(self.fc2(se))
        se = torch.permute(se, (0,2,1))

        x = self.act3(x * se) 
        x = x.permute(0,2,1)

        return x

'''
Build the GNN
'''

class gxnet(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets.
    """
    def __init__(self, sta_num, input_dim, hidden_dim, output_dim, depth, order, given_A_q, given_A_h, min_lim=None):
        super(gxnet, self).__init__()
        self.channel = sta_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.order = order
        self.given_A_q = given_A_q
        self.given_A_h = given_A_h
        self.min_lim = min_lim

        # Adjust the A_q and A_h
        #self.given_A_q = 1-self.given_A_q
        #self.given_A_h = 1-self.given_A_h
        #self.given_A_q = self.given_A_q/self.given_A_q.sum(axis=0)
        #self.given_A_h = self.given_A_h/self.given_A_h.sum(axis=0)

        self.GNN_begin = D_GCN(self.input_dim, self.hidden_dim, self.order)
        self.blocklist = nn.ModuleList([ResBlock(self.channel, self.hidden_dim, self.order) for i in range(self.depth)])
        self.GNN_end = D_GCN(self.hidden_dim, self.output_dim, self.order, activation = 'linear')

        self.adjust_A_q = nn.Parameter(self.given_A_q, requires_grad=True)
        self.adjust_A_h = nn.Parameter(self.given_A_h, requires_grad=True)

        self.act = nn.SiLU()
        self.As_act = nn.ReLU()

    def forward(self, x):
        """
        :param x: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed x of shape (batch_size, num_timesteps, num_nodes)
        """
        # Process the random walk matrix
        adjust_A_q = self.As_act(self.adjust_A_q) # A_q and A_h shall be non-negative
        adjust_A_h = self.As_act(self.adjust_A_h)
        adjust_A_q = adjust_A_q/adjust_A_q.sum(axis=0) # A_q.sum(axis=0) shall be 1
        adjust_A_h = adjust_A_h/adjust_A_h.sum(axis=0) # So as A_h

        x = x.permute(0, 2, 1)
        x = self.GNN_begin(x, adjust_A_q, adjust_A_h)
        x = self.act(x)
        for block in self.blocklist:
            #shortcut = x
            x = block(x, adjust_A_q, adjust_A_h)
            #x = x + shortcut
        x = self.GNN_end(x, adjust_A_q, adjust_A_h)
        x = x.permute(0, 2, 1).squeeze()

        if self.min_lim is not None:
            x = torch.clamp(x, self.min_lim)

        return x

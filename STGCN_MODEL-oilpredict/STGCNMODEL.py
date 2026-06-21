import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class GCN_model(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN_model, self).__init__()
        torch.manual_seed(520)
        self.num_features = num_features                #  节点的数量
        self.num_classes = num_classes                  #  输出的通道
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.conv1 = GCNConv(self.num_features, 4)  # 只定义子输入特证和输出特证即可
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.classifier = Linear(2, self.num_classes)



    def forward(self, x, edge_index):

        t = self.temporal1(X)
        # 3层GCN
        h = self.convl(x, edge_index)  # 给入特征与邻接矩阵（注意格式，上面那种）
        h = h.tanh()
        t3 = self.temporal2(t2)
        # 分类层
        out = self.classifier(h)
        return out, h
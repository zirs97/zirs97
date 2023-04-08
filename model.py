import torch
import torch.nn as nn
import torch.nn.functional as F



class Graph_Learn(nn.Module):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    #batch_size ，时序，通道数量（节点数量），特征
    def __init__(self, alpha):
        super(Graph_Learn, self).__init__()
        self.alpha = alpha
        self.S = torch.tensor([[[0.0]]]) #S是图邻接矩阵
        self.diff = torch.tensor([[[[0.0]]]]) #表示节点之间的功能差异
    def build(self, input_shape): #建立一个图
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = nn.Parameter(torch.Tensor(num_of_features, 1))
        nn.init.uniform_(self.a)
        super(Graph_Learn, self).build(input_shape)

    def forward(self, x):
        _, T, V, F = x.shape
        N = x.shape[0]

        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            # shape: (N,V,F) use the current slice
            xt = x[:, time_step, :, :]
            # shape: (N,V,V)
            diff = (xt.unsqueeze(2) - xt.unsqueeze(1)).abs() #计算两个节点之间的差异
            # shape: (N,V,V)
            tmpS = torch.exp(torch.matmul(diff.transpose(1, 0), self.a).view(N, V, V))
            # normalization
            S = tmpS / tmpS.sum(dim=1, keepdim=True).repeat(1, V, 1)

            diff_tmp += diff.abs()
            outputs.append(S)
        outputs = torch.stack(outputs, dim=1)
        self.S = torch.mean(outputs, dim=1)
        self.diff = torch.mean(diff_tmp, dim=0) / T
        return outputs

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices,num_of_vertices, num_of_vertices)
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[2])
    
#训练FC model

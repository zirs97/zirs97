import torch
import torch.nn as nn
import torch.nn.functional as F
class FCGraph(nn.Module):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, alpha):
        super(FCGraph, self).__init__()
        self.alpha = alpha
        self.S = torch.tensor([[[0.0]]]).double()  # similar to placeholder
        self.diff = torch.tensor([[[[0.0]]]]).double() # similar to placeholder
        # nn.Parameter 是一种特殊的 Tensor 类型，用于表示模型的可学习参数。通过将 Tensor 对象转换为 nn.Parameter 对象，
        # 可以指示 PyTorch 将这些参数视为模型的一部分，并在训练过程中对它们进行优化。
        self.a = nn.Parameter(torch.randn((5, 1)).double())
        nn.init.uniform_(self.a) #初始化权重
        #self.bn = nn.BatchNorm1d()

    def forward(self, x):
        N, T, V, F = x.size()
        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            ##(N,V,F)
            #计算每个时间步中顶点之间的特征差异
            #(N,V,1,F) - (N,V,F) =(N,V,V,F)
            #pytorch中是有自动广播的功能的
            #xt.unsqueeze(2) (N,V,1,F)
            #xt.unsqueeze(1)  (N,1,V,F)
            # 相减后 diff (N,V,V,F)
            xt = x[:, time_step, :, :] # (32, 12, 5) (N, V, F)
            # shape: (N,V,V)
            xt1 = xt.transpose(0,1) #(V,N,F)  (12, 32, 5)
            # shape: (V,N,V,F)
            x2 = xt1.unsqueeze(0) #(1, 12, 32, 5)
            x2 = x2.permute(0,2,1,3) #(1, 32, 12, 5)
           #(V, N, V, F) 
            diff = xt1.unsqueeze(2) - x2 #(V, N, 1, F) - (1, N, V, F)  #现在已经transpose到(V, N, V, F) 点乘a a.shape = (F, 1) 乘完后的shape应该是(V, N, V, 1)
        
            diff_a = torch.matmul(torch.transpose(torch.abs(diff), 1, 0), self.a) #将diff先取绝对值 然后转换成二维矩阵才能与a做点乘
            diff_re = diff_a.reshape(N,V,V)
            M ,_= torch.max(diff_re, dim=1, keepdim=True) #第二个返回值时最大值的下标 (32, 12, 12)
            m ,_=torch.min(diff_re,dim=1, keepdim=True)
            diff_new = (diff_re - M)/(M-m)
            
            #diff_re = self.bn(diff_re)
            tmpS = torch.exp(diff_new) #(N,V,V) (32, 12, 12)
            sum = torch.sum(tmpS, dim=1, keepdim=True) # (32, 1, 12)
            # log_S = diff_new - torch.log(torch.sum(torch.exp(diff_new),dim=1,keepdim=True))
            # re_S = torch.exp(log_S)
            # normalization 
            S = tmpS/sum #(32, 12, 12) 进行了自动广播 (N, V, V)
            #S = S.permute(1, 2, 0) #(N, V, V)

            diff_tmp += torch.abs(diff) #(12, 32, 12, 5)
            outputs.append(S)
    
        outputs = torch.transpose(torch.stack(outputs), 0, 1) #(32, 8064, 12, 12)
        # Output: (batch_size, num_of_vertices, num_of_vertices)
        self.S = torch.mean(outputs, dim=0) #(8064, 12, 12)
        self.diff = torch.mean(diff_tmp, dim=1) / T #( 12, 12, 5)
        #返回一个新的张量，新张量与原张量有相同的数值，但是不再与计算图相连。也就是说，对新张量的操作不会对原来的计算图产生影响
        return outputs, self.S, self.diff
    # def extra_repr(self):
    #     return 'alpha={}'.format(self.alpha)

class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
#(8064, 12, 12)
#( 12, 12, 5)
    def forward(self, diff, S):
        if len(S.shape) == 4:
            return torch.mean(torch.sum(torch.sum(diff**2, dim=3)*S, dim=(1,2)))
        else: #diff.shape= (12, 12, 5)
            #torch.sum(diff**2, dim=2)  shape = (12, 12)
            return torch.sum(torch.sum(diff**2, dim=2)*S) #(12, 12)^2 * (8064, 12, 12)
        
class FNormLoss(nn.Module):
    def __init__(self, Falpha):
        super(FNormLoss, self).__init__()
        self.Falpha = Falpha

    def forward(self, S):
        if len(S.shape) == 4:
            return self.Falpha*torch.sum(torch.mean(S**2, dim=0))
        else:
            return self.Falpha*torch.sum(S**2)
        
 #正则化系数


# x = torch.randn(size=(32, 8064, 12, 5))
#     # Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features) #(32, 8064, 12,5)
#     # Output: (batch_size, num_of_vertices, num_of_vertices)
# a = torch.randn(size=(5,1)) #shape=(num_of_features, 1),
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset): #(1280, 8064, 12, 5)
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        d = self.data[index,:,:,:]
        return torch.Tensor(d)
    def __len__(self):
        return(self.data.shape[0])

def train_gen(train_loader, alpha,num_epochs):
    model = FCGraph(alpha)
    #model = model.float() 单精度 不够
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    diff_loss = DiffLoss()
    fnorm_loss = FNormLoss(Falpha=0.01)
    for i in range(num_epochs):
        model.train()
        train_loss = 0
        for iter,train_set in enumerate(train_loader):
            # Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
            data = train_set #(32, 8064, 12, 5)
            output, S, diff = model(data) #(8064, 12, 12) (32, 12, 5) 
            loss = diff_loss(diff, S) + fnorm_loss(S)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if i%10 == 0:
            print('epoch: %d,train_loss: %.3f'%(i,train_loss/len(train_set)))
        return train_loss/len(train_set)

import numpy as np
data = np.load('/Users/zirs/Desktop/SAE/data/data_gengraph.npy') #(1280, 8064, 12, 5)
data = torch.from_numpy(data)
dataset = MyDataset(data)
train_loader = DataLoader(dataset, 32, shuffle=True)
train_gen(train_loader,alpha=0.0001,num_epochs=100)
# for time_step in range(T):
#     xt = x[:,time_step,:,:] ##(N,V,F)
#     #计算每个时间步中顶点之间的特征差异
#     #(N,V,1,F) - (N,V,F) =(N,V,V,F)
#     #pytorch中是有自动广播的功能的
#     #xt.unsqueeze(2) (N,V,1,F)
#     #xt.unsqueeze(1)  (N,1,V,F)
#     # 相减后 diff (N,V,V,F)
#     diff = xt.unsqueeze(2) - xt.unsqueeze(1) #(N, V, 1, F) 减 (N, 1, V, F) 当张量的某些维度长度为 1 时，可以将它们沿着该维度进行复制，直到长度与另一个张量相等。
#     #转置后 (N,V,V,F) 两个节点交换位置 然后n与V再交换位置
#     diff = diff.permute(0,2,1,3) # (N,V,V,F)
    
#     diff_tmp +=diff.abs().sum(dim=-1)
#     #计算邻接矩阵 (N,V,V)
#     mid_m = diff.abs().matmul(a)
#     tmpS = torch.exp((diff.abs().matmul(a))) 
#     tmpS1 = tmpS.reshape((32,12,12))
#     tmpS2 = tmpS1.sum(dim=1, keepdim=True)
#     tmpS3 = tmpS2.reshape((12,32,12))
#     S = tmpS / tmpS.sum(dim=1, keepdim=True)
#     outputs.append(S)

# outputs = torch.stack(outputs, dim=1)  # (N, T, V, V) 在第一维度按时间堆积
# S = outputs.mean(dim=1)  # (N, V, V)
# diff = diff_tmp.mean(dim=1) / T  # (N, V, V)
# #返回一个新的张量，新张量与原张量有相同的数值，但是不再与计算图相连。也就是说，对新张量的操作不会对原来的计算图产生影响
# diff = diff.detach()

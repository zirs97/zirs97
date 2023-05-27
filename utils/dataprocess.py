# 处理数据和标签
import numpy as np
from torch.utils.data import DataLoader,Dataset
import os
import _pickle as pickle
import mne
import torch
# 数据标签加载
# data的shape = (num_samples, num_channels, 时序) label.shape = (num_samples, )
pwd = os.getcwd() #获取当前工作路径
label= np.load(pwd+'/data/label1_s.npy')
data = np.load(pwd+'/data/data_s12.npy') # (1280, 8064, 12)
data = np.swapaxes(data, 1,2) #(1280, 12, 8064)

# 处理raw数据
class raw_process():
    def __init__(self,dir,num_sub=32,freq=128,drop=True,window=True,window_size=10,step_size=10,norm=True,num_class=3,save=True):
        self.dir = dir
        self.num_sub = num_sub
        self.freq = freq
        self.drop = drop
        self.window = window
        self.window_size = window_size
        self.step_size = step_size
        self.time_step = int(self.window_size*self.freq) # 10*1280
        self.norm = norm
        self.num_class = num_class
        self.save = save

    # 读取数据
    def read(self):
        data_l, labels_val_l, labels_aro_l = [],[],[]
        for sub in range(1,1+self.num_sub):
            with open(self.dir+'/s'+str(sub).rjust(2,'0')+'.dat','rb') as file:
                subject = pickle.load(file, encoding='latin1')
                data, labels = subject['data'], subject['labels']
                data = data[:,0:32,:] # 取前32个通道 (40, 32, 8064)
                if self.drop==True:
                    data = data[:,:,3*self.freq:] # 丢弃前3s的数据 (40, 32, 7680)
                labels_val,labels_aro = labels[:,0], labels[:,1] # (40,)
                # 滑动窗口取数据
                if self.window==True:
                    data,label = self.sliding_window(data=data,labels=labels_val) # (240 ,32 ,1280) (240,)
                data_l.append(data)
                labels_val_l.append(label)
        data_all = np.array(data_l).reshape((-1,32,self.time_step)) #(32*240, 32, 1280)
        if self.norm:
            data_all = self.scale_data(data_all) # 时间维度归一化
        labels_val_all = np.array(labels_val_l).flatten()
        labels_val_all = self.label_pre(labels_val_all,self.num_class) # 将SAM量表转化为标签
        if self.save == True:
            np.save('./data/data.npy',data_all)
            np.save('./data/label_val.npy',labels_val_all)
        return data_all,labels_val_all
                

    def sliding_window(self,data,labels):
        labels = labels.flatten() # (40,)
        num = int(data.shape[2]/self.time_step) # 6
        sub_data = np.zeros((len(data),num,32,self.time_step)) # (40, 6, 32, 1280)
        sub_label = np.zeros((len(data),num)) # (40, 6)
        for j in range(len(data)): # 40个视频
            data1 = data[j,:,:] # (32, 7680)
            label1 = labels[j] #(1,)
            # for i in range(0, data1.shape[1]-window_size+1, step_size):
            # for iter,i in enumerate(range(0, data1.shape[1]-self.window_size+1, self.step_size)):
            for i in range(num):
                segment = data1[:,i*self.time_step:(i+1)*self.time_step] # (32, 1280)
                sub_data[j,i,:,:] = segment
                sub_label[j,i] = label1
        sub_data,sub_label = sub_data.reshape((-1,sub_data.shape[2],sub_data.shape[3])), sub_label.flatten()
        return sub_data,sub_label # (240 ,32 ,1280) (240,)
    
    # 将SAM量表转化为标签
    def label_pre(self,label, num_class:int):
        """
        Parameters
        -----------
        label : narray
            label.shape = (样本数, 1)
        num_class : int
            2/3
        dir : str
        """
        if num_class == 2:
            for i in range(len(label)):
                y = label[i]
                if y <=5:
                    label[i] = 0
                else:
                    label[i] = 1
        elif num_class == 3:
            for i in range(len(label)):
                y = label[i]
                if y <=3:
                    label[i] = 0
                elif y<=6:
                    label[i] = 1
                else:
                    label[i] = 2
        else:
            print('no options!')
        label = label.astype(int) 
        return label.flatten()
    
    def scale_data(self,data):
        '''
        data.shape = (n_samplers, n_channels, time_step)
        '''
        mean = np.mean(data, axis=(0,1), keepdims=True)
        std = np.std(data, axis=(0,1), keepdims=True)
        data_norm = (data - mean)/std
        return data_norm

    
process = raw_process('/Users/zirs/data/DEAP/data_preprocessed_python')
# x = np.load('./data/data.npy')
# data = process.scale_data(x)
# np.save('./data/data_norm.npy',data)
# process.read()

# 生成dataset类
class MyDataset(Dataset):
    '''
    parameters:
    -----------
    data: (num_samples, num_channels, time_steps)
    labels: (num_samples)
    '''
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        d, l = self.data[idx], self.labels[idx] #将numpy数组转换为整数 否则为array([1.])
        return d, l
    
    def __len__(self):
        return len(self.data)

# k折交叉验证
from sklearn.model_selection import KFold
def K_fold(n_splits,data,label,batch_size,fold):
    dataset = MyDataset(data, label)
    kf = KFold(n_splits=n_splits, shuffle=True)
    for i,(train_index,val_index) in enumerate(kf.split(dataset)): # k.split返回的是有索引值组成的数组
        train_,val_ = [dataset[i] for i in train_index], [dataset[i] for i in val_index] #由data组成的列表 data= X,y
        train_data, train_label = [train_[i][0] for i in range(len(train_))], [train_[i][1] for i in range(len(train_))]
        val_data, val_label = [val_[i][0] for i in range(len(val_))], [val_[i][1] for i in range(len(val_))]
        train_set,val_set = MyDataset(train_data,train_label),MyDataset(val_data,val_label)
        train_loader,val_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True),DataLoader(val_set,batch_size=batch_size,shuffle=False)
        if fold==i:
            return train_loader,val_loader

# 随机划分数据集
import random
import torch
def split_data(data,label, a):
    dataset = MyDataset(data,label)
    train_size = int(len(dataset)*a)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset,[train_size, test_size])
    return train_set,test_set

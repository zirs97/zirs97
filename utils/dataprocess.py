# 处理数据和标签
import numpy as np
from torch.utils.data import DataLoader,Dataset
import os

# 数据标签加载
# data的shape = (num_samples, num_channels, 时序) label.shape = (num_samples, )
pwd = os.getcwd() #获取当前工作路径
label= np.load(pwd+'/data/label1_s.npy')
data = np.load(pwd+'/data/data_s12.npy') # (1280, 8064, 12)
data = np.swapaxes(data, 1,2) #(1280, 12, 8064)

# 将SAM量表转化为标签
def label_pre(label, num_class:int, dir:str):
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
    label = label.astype(int) #(1280, 1)
    label = label.flatten() #(1280,)
    np.save(dir, label)

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
def K_fold(n_splits,data,label,batch_size):
    dataset = MyDataset(data, label)
    kf = KFold(n_splits=n_splits, shuffle=True)
    for fold,(train_index,val_index) in enumerate(kf.split(dataset)): # k.split返回的是有索引值组成的数组
        train_,val_ = [dataset[i] for i in train_index], [dataset[i] for i in val_index] #由data组成的列表 data= X,y
        train_data, train_label = [train_[i][0] for i in range(len(train_))], [train_[i][1] for i in range(len(train_))]
        val_data, val_label = [val_[i][0] for i in range(len(val_))], [val_[i][1] for i in range(len(val_))]
        train_set,val_set = MyDataset(train_data,train_label),MyDataset(val_data,val_label)
        train_loader,val_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True),DataLoader(val_set,batch_size=batch_size,shuffle=False)
    return train_loader,val_loader

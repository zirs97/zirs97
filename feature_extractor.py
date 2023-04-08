# feature_extractor
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self,num_class, channels=12, time_second = 63, freq = 128 ):
        # 输入数据shape 每个受试者的每个视频作为一条数据输入 （通道数， 时序， 1） (32, 8024, 1)
        super(FeatureExtractor, self).__init__()
    ######### CNNs with small filter size at the first layer 
    # Small convolution kernel is better at capturing temporal information
        # self.input_signal = nn.Sequential(
        #         nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=50, stride=6),
        #         nn.BatchNorm1d(32),
        #         nn.ReLU()
        #     )

        self.cnn_s = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=32,
                      kernel_size=50,
                      stride=6,
                      padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16,stride=16),
            nn.Dropout(p=0.5)
        )
        self.cnn_s1 = nn.Sequential(
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=8,
                      stride=1,
                      padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5) #自己后来加的
        )
        self.cnn_s2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=8,
                      stride=1,
                      padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5) #自己后来加的
        )
        self.cnn_s3 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=8,
                      stride=1,
                      padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5) #自己后来加的
        )
        self.cnn_sMF = nn.Sequential(
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Flatten()
        )
  ######### CNNs with large filter size at the first layer #########
  #large convolution kernel is better at capturing frequency information
        self.cnn_l = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=64,
                      kernel_size=400,
                      stride=50,
                      padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8,stride=8),
            nn.Dropout(p=0.5)
        )
        self.cnn_l1 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=6,
                      stride=1,
                      padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5) # 自己后来加的
        )
        self.cnn_l2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=6,
                      stride=1,
                      padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5) #自己后来加的
        )
        self.cnn_l3 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=6,
                      stride=1,
                      padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5) #自己后来加的
        )
        self.cnn_lMF = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten()
        )
        #这一层的目的是 把得到的特征输入进去 得出对标签的预测 从而确定学习到的特征是有效的
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=896,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_class),
            # nn.Softmax(dim=1)
        )
    def forward(self, x):
        # x = self.input_signal(x)
        x = x.to(torch.float32) #(32,32,8064)
        s = self.cnn_s(x) #(32,640)
        s = self.cnn_s1(s)
        s = self.cnn_s2(s)
        s = self.cnn_s3(s)
        s = self.cnn_sMF(s)
        l = self.cnn_l(x) #(32,256)
        l = self.cnn_l1(l)
        l = self.cnn_l2(l)
        l = self.cnn_l3(l)
        l = self.cnn_lMF(l)
        feature=torch.cat([s,l],dim=1) #(32,896)
        out = self.fc(feature)
        return out, feature #out是分类结果 feature是学习到的特征


## k-fold cross validation 加载数据
import numpy as np
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

# 用32通道
# data_ = np.load('/Users/zirs/Desktop/SAE/data/data_all.npy') # (32, 40, 32, 8064)= (32个受试者 ,40个视频, 32个通道, 63*128)
# data_= data_.reshape((-1,data_.shape[2],data_.shape[3]))
# data_ = data_.reshape((data_.shape[0],data_.shape[1],data_.shape[2])) #(1280, 32, 8064)

# 用12通道
data_ = np.load('/Users/zirs/Desktop/SAE/data/data_12.npy') # (1280, 8064, 12)
data_ = np.swapaxes(data_, 1,2) #(1280, 12, 8064)
# y = np.load('/Users/zirs/Desktop/SAE/data/label_val.npy') #(32 , 40, 1)
#将数据和标签组合成字典

y = np.load('/Users/zirs/Desktop/SAE/data/label.npy') #(1280,1)
y2 = np.load('/Users/zirs/Desktop/SAE/data/label_s2.npy') # 二分类
#y2 = y2.flatten()
#y =  y.reshape((-1,1)) #(1280, 1)
# # 时序数据标准化
# mean = np.mean(data_, axis=(0,2), keepdims=True)
# std = np.std(data_,axis=(0,2), keepdims=True)
# data_ = (data_-mean)/std



datalist, labellist= [],[]
for i in range(len(data_)):
    data1, label1 = data_[i,:,:], y[i]
    # if label1<=5:
    #     label1 = 0
    # else:
    #     label1 = 1
    datalist.append(data1) # 1280 * (32,8064)
    labellist.append(label1)


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# datalist = scaler.fit_transform(datalist)

from torch.utils.data import DataLoader,Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        d, l = self.data[idx], self.labels[idx] #将numpy数组转换为整数 否则为array([1.])
        #l.reshape((1))
        return d, l
    
    def __len__(self):
        return len(self.data)
    
dataset = MyDataset(datalist, labellist)
datalist2, labellist2= [],[]
for i in range(len(data_)):
    data1, label1 = data_[i,:,:], y2[i]
    # if label1<=5:
    #     label1 = 0
    # else:
    #     label1 = 1
    datalist2.append(data1) # 1280 * (32,8064)
    labellist2.append(label1)

dataset_s2 = MyDataset(datalist2, labellist2)



import torch.utils.data as data_utils
from torch.utils.data import DataLoader
# train_r,val_r= 0.8, 0.2# 训练集占比 这样1536乘出来才是整数 1152 192 192
# train_size, val_size = int(train_r * len(dataset)), int(val_r*len(dataset))
# train_dataset, val_dataset= data_utils.random_split(datalist, [train_size, val_size])

def save_best_model(model, optimizer, val_acc,save_path):
    best_val_acc = getattr(save_best_model,'best_val_acc', None) #相当于save_best_model.best_val_acc
    if best_val_acc is None or val_acc> best_val_acc:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), #保存神经网络优化器状态的字典。
            'val_acc': val_acc
        },save_path)

def eval_acc(val_loader, model):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for iter,data in enumerate(val_loader):
            X, y = data
            pred,_ = model(X) 
            acc_sum += (pred.argmax(dim=1)==y).float().sum().item()
            n += y.shape[0]
    return acc_sum/n


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
transfer = PCA(n_components=3)
#from lightning.pytorch.callbacks import ModelCheckpoint

def train(train_loader,val_loader,num_epoch,model,criterion,optimizer):
    for i in range(num_epoch):
        model.train()
        train_acc_sum,n ,l_sum= 0.0 ,0, 0.0
        for iter,data in enumerate(train_loader):
            X,y = data # X.shape = (32,32,8064)
            pred,fea= model(X) # _为feature pred为outputs  即分类的标签
            target = torch.argmax(pred, dim=1) #得出概率最大的标签
            l = criterion(pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n+= y.shape[0]
            l_sum +=l
            train_acc_sum += (pred.argmax(1)==y).sum().item()
        val_acc =eval_acc(val_loader,model)
        if i%10 ==0:
            print('epoch %d, loss %.4f, train_acc %.3f, val_acc %.3f'%(i, l_sum/iter,train_acc_sum/n, val_acc))
        if i == 70:
            train_loader1 = DataLoader(dataset,batch_size=128,shuffle=True)
            for iter,data in enumerate(train_loader1) :
                if iter == 3:
                    pred1, fea = model(data[0])
                    feature = fea.detach().numpy()
                    target1 = torch.argmax(pred1, dim=1)
                    lab = target1.detach().numpy()
                    feature_new = transfer.fit_transform(feature)
                    x_pca=np.dot(feature_new,transfer.components_)
                    fig = plt.figure(figsize=(8,8))
                    #ax = Axes3D(fig)
                    ax = fig.add_subplot(111, projection='3d', auto_add_to_figure=False)
                    plt.grid(True)
                    # plt.title("PCA降维之后的情况")
                    # plt.xlim(-15,15)
                    # plt.ylim(-15,15)
                    # plt.zlim(-15,15)
                    x, y, z = x_pca[:,0],x_pca[:,1],x_pca[:,2]
                    ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2])
                    colors = ['r', 'b', 'g']  # 红色和蓝色分别对应两个标签
                    for i in range(len(x_pca)):
                        ax.scatter(x[i], y[i], z[i], c=colors[lab[i]])
                    # plt.show()
                    # ax.set_xlabel('feature1')
                    # ax.set_ylabel('feature2')
                    # ax.set_zlabel('feature3')
                    plt.show()




def train_fold(dataset,num_epoch, num_class):
    for fold,(train_index,val_index) in enumerate(kf.split(dataset)):
        model = FeatureExtractor(num_class=num_class)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 ,weight_decay=0.005 )  #number   weight_decay=0.001 
        criterion = torch.nn.CrossEntropyLoss()
        print('Fold:',fold,'#'*113)
        train_,val_ = [dataset[i] for i in train_index], [dataset[i] for i in val_index] #由data组成的列表 data= X,y
        train_data, train_label = [train_[i][0] for i in range(len(train_))], [train_[i][1] for i in range(len(train_))]
        val_data, val_label = [val_[i][0] for i in range(len(val_))], [val_[i][1] for i in range(len(val_))]
        train_set,val_set = MyDataset(train_data,train_label),MyDataset(val_data,val_label)
        train_loader,val_loader = DataLoader(train_set,batch_size=32,shuffle=True),DataLoader(val_set,batch_size=32,shuffle=False)
        
        train(train_loader=train_loader,val_loader=val_loader,num_epoch=num_epoch,model=model,criterion=criterion,optimizer=optimizer)
           

#train_fold(dataset=dataset_s2,num_epoch=200, num_class=2)
train_fold(dataset=dataset,num_epoch=200, num_class=3)

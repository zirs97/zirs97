import numpy as np
import argparse
from utils import dataprocess
from torch.utils.data import DataLoader
from utils.dataprocess import MyDataset,K_fold,split_data
from utils.misc import Averager
import torch
from models.SAE import MySAE,MySAE1
import os
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import shutil,sys
class SAETrainer(object):
    def __init__(self,args):
        self.args = args
        self.base_path = self.args.base_path
        self.base_save_path = './runs'
        self.data, self.label = np.load('./data/data_norm.npy'), np.load('./data/label_val.npy')
        self.train_set, self.test_set = split_data(self.data,self.label,self.args.a)
        self.train_loader, self.test_loader = DataLoader(self.train_set,self.args.batch_size,shuffle=True), DataLoader(self.test_set,self.args.batch_size,shuffle=False)
        self.model = MySAE1()
        self.loss = torch.nn.MSELoss()

    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), os.path.join(self.base_save_path, name+'.pth'))
    
    def train_fold(self):
        for fold in range(self.args.n_splits):
            train_loader, val_loader = K_fold(n_splits=self.args.n_splits,data=self.data,label=self.label,batch_size=self.args.batch_size,fold=fold)
            print('Fold:',fold+1,'#'*113)
            model = self.model # 将模型初始化为空模型
            optimizer = torch.optim.Adam(model.parameters(),lr=self.args.lr)
            for epoch in range(1,self.args.num_epochs+1):
                model.train()
                train_loss_averager = Averager()
                for iter,batch in enumerate(train_loader,1):
                    X,_ = batch # X.shape = (32, 32, 1280) (batch_size, num_channels, time_step)
                    X = X.permute(0,2,1).to(torch.float) # (batch_size, time_step, num_channels)
                    encode, rec_x = model(X)
                    l = self.loss(rec_x,X)  # 重构X与源数据的差值
                    train_loss_averager.add(l.item()) # 加入新的loss之后整体再平均
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()

                train_loss_averager = train_loss_averager.item()

                model.eval()
                val_loss_averager = Averager()
                for iter,batch in enumerate(val_loader,1):
                    X,_ = batch
                    X = X.permute(0,2,1).to(torch.float)
                    _,rec_x = model(X)
                    l = self.loss(rec_x,X)
                    val_loss_averager.add(l.item())
                val_loss_averager = val_loss_averager.item()

                if epoch%5 == 0 :      
                    # print('Epoch {}, Train: Loss={:.4f} Acc={:.4f} Val: Loss={:.4f} Acc={:.4f}'.format(epoch, train_loss_averager,train_acc_averager,val_loss_averager, val_acc_averager))
                    print('Epoch {}, Train: Loss={:.4f} Val: Loss={:.4f}'.format(epoch, train_loss_averager,val_loss_averager))

    def train_layers(self,layers_list=None, layer=None, validate=True):
        print('>>start training the %s layer'%(str(layer+1)))
        train_loader, test_loader = self.train_loader,self.test_loader
        optimizer = torch.optim.Adam(layers_list[layer].parameters(),lr=self.args.lr_layers,weight_decay=self.args.wd_layers)
        # optimizer = torch.optim.SGD(layers_list[layer].parameters(),lr=self.args.lr)
        # writer = SummaryWriter()
        for epoch in range(1,self.args.num_epochs_layers+1):
            sum_loss = 0.
            train_loss_averager = Averager()
            if layer != 0:
                for index in range(layer):
                    layers_list[index].lock_grad() # 冻结更新
                    layers_list[index].is_training_layer = False # 冻结输出的返回方式

            for iter, (X,_) in enumerate(train_loader):
                out = X.permute(0,2,1).to(torch.float)
                if layer != 0:
                    for i in range(layer):
                        out = layers_list[i](out) # 对前(layer-1)冻结了的层进行前向计算
            
                pred = layers_list[layer](out)
                optimizer.zero_grad()
                l = self.loss(pred, out)
                train_loss_averager.add(l.item()) # 加入新的loss之后整体再平均
                l.backward()
                optimizer.step()
            train_loss_averager = train_loss_averager.item() 
            writer.add_scalar('Loss_layers/'+str(layer+1),train_loss_averager,epoch)
            if validate:
                val_loss_averager = Averager()
                for iter, (X,_) in enumerate(test_loader):
                    out = X.permute(0,2,1).to(torch.float)
                    if layer != 0:
                        for i in range(layer):
                            out = layers_list[i](out) # 对前(layer-1)冻结了的层进行前向计算
                
                    pred = layers_list[layer](out)
                    l = self.loss(pred, out)
                    val_loss_averager.add(l.item()) # 加入新的loss之后整体再平均
                val_loss_averager = val_loss_averager.item()
            if epoch%5 == 0 :      
                print('Epoch {}, Train: Loss={:.4f} Val: Loss={:.4f}'.format(epoch, train_loss_averager,val_loss_averager))

    def train_whole(self,model=None,validate=True):
        print('>>start training whole model')
        # 解锁冻结的参数
        for param in model.parameters():
            param.require_grad = True
        optimizer = torch.optim.Adam([{'params':model.parameters(),'initial_lr':self.args.lr_whole}],lr=self.args.lr_whole,weight_decay=self.args.wd_whole)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.7, last_epoch=150)
        for epoch in range(1,self.args.num_epochs_whole+1):
            train_loss_averager = Averager()
            for iter, (X, _) in enumerate(self.train_loader):
                X = X.permute(0,2,1).to(torch.float)
                out = model(X) # out是encoder的输出 没有经过decoder
                optimizer.zero_grad()
                l = self.loss(out, X)
                train_loss_averager.add(l.item()) # 加入新的loss之后整体再平均
                l.backward()
                optimizer.step()
            # scheduler.step()
            train_loss_averager = train_loss_averager.item()
            writer.add_scalar('Loss_whole/train',train_loss_averager,epoch)
            if validate:
                val_loss_averager = Averager()
                for iter,(X,_) in enumerate(self.test_loader):
                    X = X.permute(0,2,1).to(torch.float)
                    out = model(X)
                    l = self.loss(out, X)
                    val_loss_averager.add(l.item())
                val_loss_averager = val_loss_averager.item()
                writer.add_scalar('Loss_whole/val',val_loss_averager,epoch)
            if epoch%10 == 0 :      
                print('Epoch {}, Train: Loss={:.4f} Val: Loss={:.4f}'.format(epoch, train_loss_averager,val_loss_averager))
        writer.close()
        print('<<end training whole model')

from models.SAE import AutoEncoderLayer,SAE

def stack_model(h1=64,h2=12):
    encoder1 = AutoEncoderLayer(input_dim=32,output_dim=h1,SelfTraining=True)
    encoder2 = AutoEncoderLayer(input_dim=h1,output_dim=h2,SelfTraining=True)
    decoder3 = AutoEncoderLayer(input_dim=h2,output_dim=h1,SelfTraining=True)
    decoder4 = AutoEncoderLayer(input_dim=h1,output_dim=32,SelfTraining=True)
    layers_list = [encoder1, encoder2, decoder3, decoder4]
    return layers_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path',type=str,default='')
    # 训练参数
    parser.add_argument('--train_layers', type=bool,default=True) # 是否进行分层训练
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--a',type=float,default=0.8) # 训练测试集划分比例
    parser.add_argument('--lr_layers',type=float,default=1e-3)
    parser.add_argument('--wd_layers',type=float,default=0)
    parser.add_argument('--lr_whole',type=float,default=1e-3)
    parser.add_argument('--wd_whole',type=float,default=0)
    parser.add_argument('--num_epochs_layers', type=int, default=100)
    parser.add_argument('--num_epochs_whole', type=int, default=500)
    args = parser.parse_args()

    trainer = SAETrainer(args)
    train_time = datetime.fromtimestamp(int(time.time())).strftime("%m-%d.%H:%M")
    try:
        layers_list = stack_model()
        # 输出方式1:经过encoder decoder之后输出 用于分层训练
        # 输出方式2:只经过encoder输出 用于统一训练
        log_dir='./runs/'+train_time
        writer = SummaryWriter(log_dir=log_dir) # 写入tensorboard
        if args.train_layers:
            # 分层训练
            for level in range(len(layers_list)):
                trainer.train_layers(layers_list=layers_list,layer=level)
        # 统一训练
        SAE_model = SAE(layers_list=layers_list)
        trainer.train_whole(model=SAE_model)
        torch.save(SAE_model,'./runs/models/SAE_model.pt')
    except KeyboardInterrupt:
        writer.close()
        # shutil.rmtree(log_dir)
        # os.remove(log_dir)
        print(">>end process")
        # raise KeyboardInterrupt
        sys.exit(0)
    # trainer.train_fold()

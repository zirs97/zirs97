import numpy as np
import torch
import os 
import torch.nn.functional as F
from utils.misc import Averager
from torch.utils.tensorboard import SummaryWriter
import time
import tqdm
from utils.dataprocess import K_fold
from utils.misc import Timer
from models.feature_extractor import FeatureExtractor

class FEtrainer(object):
    def __init__(self, args):
        # code for args
        self.args = args
        if not os.path.exists(args.base_save_path):
            os.makedirs(args.base_save_path)
        self.outputs_eval_file = os.path.join(args.base_save_path, time.strftime("%m%d%H%M")+"eval_results.txt")
        self.sum_outputs = os.path.join(args.base_save_path,time.strftime("%b%d%H%M")+'.txt')
        print("perparing dataset loader")
        self.num_class = self.args.num_class
        # read data and process data
        self.data_path = args.data_path
        self.data, self.label = np.load(self.data_path+self.args.data_dir), np.load(self.data_path+self.args.label_dir)
        # 将data的维度从(1280, 8064, 12)转换成(1280, 12, 8064)
        self.data = np.swapaxes(self.data, 1,2) 
        self.train_loader, self.val_loader = K_fold(n_splits=self.args.n_splits,data=self.data,label=self.label,batch_size=self.args.batch_size)
        self.model = FeatureExtractor(self.args.num_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), os.path.join(self.args.base_save_path, name+'.pth'))
    
    def train(self):
        # 设置训练日志 是个字典
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        
        timer = Timer()
        global_count = 0
        writer = SummaryWriter(comment=self.args.base_save_path)
        train_args = {
            'num_epochs': self.args.num_epochs,
            'num_class': self.num_class,
            'lr':self.args.lr,
            'batch_size':self.args.batch_size,
        }
        print('-'*80)
        print('num_epochs:%d num_class:%d lr:%.6f batch_size:%d'%(train_args['num_epochs'],train_args['num_class'],train_args['lr'],train_args['batch_size']))
        print('-'*80)
        for epoch in range(1,self.args.num_epochs+1):
            start_time = time.time()
            self.model.train()
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            # tqdm_gen = tqdm.tqdm(self.train_loader)
            for iter,batch in enumerate(self.train_loader,1): #遍历tqdm_gen 从1开始 batch是一个批次的训练数据
                # Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
                global_count = global_count + 1
                X,y = batch #(32, 8064, 12, 5)
                pred, fea = self.model(X)
                target = torch.argmax(pred,dim=1)
                l = self.loss(pred, y)
                acc = (target==y).type(torch.FloatTensor).mean().item()
                writer.add_scalar('data/train_loss',float(l),global_count) 
                writer.add_scalar('data/train_acc', float(acc), global_count)
                train_loss_averager.add(l.item()) # 加入新的loss之后整体再平均
                train_acc_averager.add(acc)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
            
            # tqdm_10 = tqdm.tqdm(total=10)
            train_loss_averager = train_loss_averager.item() # 返回计算的平均值
            train_acc_averager = train_acc_averager.item()
            # print("--- %.4f seconds ---" % (time.time() - start_time))
            
            # 开始验证模式
            self.model.eval()
            # 打印最好的模型结果
            if epoch % 10 == 0:
                print('-'*128)
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
                #tqdm_10.update(10)

            val_loss_averager = Averager()
            val_acc_averager = Averager()
            for iter,batch in enumerate(self.val_loader,1):
                X, y = batch
                pred,fea = self.model(X)
                target = torch.argmax(pred , dim=1)
                # l = self.loss(pred,y)
                val_acc = (target==y).type(torch.FloatTensor).mean().item()
                # val_loss_averager.add(l.item())
                val_acc_averager.add(val_acc)
            
            # val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()

            # writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch) 
            if epoch%10 == 0 :      
                # print('Epoch {}, Train: Loss={:.4f} Acc={:.4f} Val: Loss={:.4f} Acc={:.4f}'.format(epoch, train_loss_averager,train_acc_averager,val_loss_averager, val_acc_averager))
                print('Epoch {}, Train: Loss={:.4f} Acc={:.4f} Val: Acc={:.4f}'.format(epoch, train_loss_averager,train_acc_averager,val_acc_averager))

    
            # 保存最好的模型
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            
            # if epoch % 10 ==0:
            #     self.save_model('epoch'+str(epoch))

            # 更新trlog 每次训练会被复写 可以作为可视化的源数据
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            torch.save(trlog, os.path.join(self.args.base_save_path, 'trlog'))

            # if epoch % 10 == 0:
            #     print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
            
            # 每个epoch的数据都保存
            # result = {'epoch': epoch,
            #          'train_loss': train_loss_averager,
            #          'train_acc':  train_acc_averager,
            #          'val_loss':   val_loss_averager,
            #          'val_acc':    val_acc_averager}   
            # with open(self.outputs_eval_file, 'a') as f :
            #     for key in result.keys():
            #         f.write("%s = %s\n"%(key, str(result[key])))
            #     f.write("-"*80)
            #     f.write('\n')
        writer.close()
        train_result = {
            'max_val_acc': trlog['max_acc'],
            'max_acc_epoch':trlog['max_acc_epoch'],
        } 
        with open(self.sum_outputs,'a') as f:
            for key in train_args.keys():
                f.write("%s = %s\n"%(key, str(train_args[key])))
            f.write("-"*80)
            f.write('\n')
            for key in train_result.keys():
                f.write("%s = %s\n"%(key, str(train_result[key])))
            # with open 会自动关闭



            
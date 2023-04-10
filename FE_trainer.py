import numpy as np
import torch
import os 
import torch.nn.functional as F
from utils.misc import Averager
from torch.utils.tensorboard import SummaryWriter
import time
import tqdm
from utils.dataprocess import K_fold

class FEtrainer(object):
    def __init__(self, args):
        self.args = args
        args.save_path = os.path.join(self.args.base_save_dir,)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        self.outputs_eval_file = os.path.join(args.save_path, time.shrftime("%m%d%H%M")+"eval_results.txt")

        print("perparing dataset loader")
        self.train_loader = args.trainloader
        self.val_loader = args.valloader
        self.model = args.model
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        self.loss = torch.nn.CrossEntropyLoss()
    

    def save_model(self, name):
        torch.save(dict(params=self.model.state_dict()), os.path.join(self.args.save_path, name+'.pth'))
    
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
        
        global_count = 0
        writer = SummaryWriter(comment=self.args.save_path)
        for epoch in range(1,self.args.num_epochs+1):
            start_time = time.time()
            self.model.train()
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for iter,batch in enumerate(tqdm_gen,1): #遍历tqdm_gen 从1开始 batch是一个批次的训练数据
                # Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
                global_count = global_count + 1
                X,y = batch #(32, 8064, 12, 5)
                pred, fea = self.model(X)
                target = torch.argmax(pred,dim=1)
                l = self.loss(pred, y)
                acc = (pred.argmax(1)==y).sum().item()
                writer.add_scalar('data/loss',float(l),global_count) 
                writer.add_scalar('data/acc', float(acc), global_count)
                train_loss_averager.add(l.item()) # 加入新的loss之后整体再平均
                train_acc_averager.add(acc)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item() # 返回计算的平均值
            train_acc_averager = train_acc_averager.item()
            print("--- %s seconds ---" % (time.time() - start_time))
            
            # 开始验证模式
            self.model.eval()
            # 打印最好的模型结果
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))

            val_loss_averager = Averager()
            val_acc_averager = Averager()
            for iter,batch in enumerate(self.val_loader,1):
                X, y = batch
                pred = self.model(X)
                target = torch.argmax(pred , dim=1)
                l = self.loss(pred,y)
                acc = (target==y).sum().item()
                val_loss_averager.add(l.item())
                val_acc_averager.add(acc)
            
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()

            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)       
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))
    
            # 保存最好的模型
            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            
            if epoch % 10 ==0:
                self.save_model('epoch'+str(epoch))

            # 更新trlog
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            torch.save(trlog, os.path.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.meta_max_epoch)))
            
            result = {'epoch': epoch,
                     'train_loss': train_loss_averager,
                     'train_acc':  train_acc_averager,
                     'val_loss':   val_loss_averager,
                     'val_acc':    val_acc_averager}   
            with open(self.outputs_eval_file, 'a') as f :
                for key in result.keys():
                    f.write("%s = %s\n"%(key, str(result[key])))
                f.write("-"*80)
                f.write('\n')
        
        writer.close()



            
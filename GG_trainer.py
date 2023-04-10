import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.misc import Averager
from torch.utils.tensorboard import SummaryWriter
import time
import tqdm


class GGtrainer(object):
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
        self.diff_loss, self.fnorm_loss = args.diff_loss, args.fnorm_loss
    

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
        write = SummaryWriter(comment=self.args.save_path)
        for epoch in range(1,self.args.num_epochs+1):
            start_time = time.time()
            self.model.train()
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for iter,batch in enumerate(tqdm_gen,1): #遍历tqdm_gen 从1开始 batch是一个批次的训练数据
                # Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
                global_count = global_count + 1
                data = batch #(32, 8064, 12, 5)
                output, S, diff = self.model(data) #(8064, 12, 12) (32, 12, 5) 
                loss = self.diff_loss(diff, S) + self.fnorm_loss(S)
                write.add_scalar('data/loss',float(loss),global_count) 
                train_loss_averager.add(loss.item()) # 加入新的loss之后整体再平均
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss_averager = train_loss_averager.item() # 返回计算的平均值
            print("--- %s seconds ---" % (time.time() - start_time))
            
            # 开始验证模式
            self.model.eval()
            # 打印最好的模型结果
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))

            val_loss_averager = Averager()
            for iter,batch in enumerate(self.val_loader,1):
                data = batch #(32, 8064, 12, 5)
                output, S, diff = self.model(data) #(8064, 12, 12) (32, 12, 5) 
                loss = self.diff_loss(diff, S) + self.fnorm_loss(S)
                val_loss_averager.add(loss.item()) # 加入新的loss之后整体再平均
    
            val_loss_averager = val_loss_averager.item()
            write.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            
            if epoch%10 == 0:
                print('epoch: %d,train_loss: %.3f'%(epoch,train_loss/len(batch)))
            return train_loss/len(batch)

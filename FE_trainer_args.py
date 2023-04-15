import argparse
from utils import dataprocess
import numpy as np
import FE_trainer
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default='data/')
    parser.add_argument('--data_dir', type=str,default='data_s12.npy',choices=['data_s12.npy','data_l12.npy'])
    parser.add_argument('--label_dir', type=str, default='label1_s3.npy', choices=['label1_s3.npy','label1_s2.npy'])
    parser.add_argument('--base_save_path', type=str, default='./runs/')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--n_splits', type=int,default=5)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=3,choices=[2,3])
   
    # set the parameters
    args = parser.parse_args() # 实例化对象

    # 开始训练
    trainer = FE_trainer.FEtrainer(args) # 将参数传入训练器 注意；这是个类 实例化
    trainer.train() # 调用函数


    
    
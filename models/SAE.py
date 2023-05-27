import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class AutoEncoderLayer(nn.Module):
    def __init__(self,input_dim, output_dim, SelfTraining=False):
        super(AutoEncoderLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training_self = SelfTraining
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features,self.out_features,bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.out_features,self.in_features,bias=True),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.encoder(x)
        if self.is_training_self: # 当为True时 解码 否则 输出的是中间层
            return self.decoder(out)
        else:
            return out # 输出中间层
    def lock_grad(self): # 冻结训练层
        for param in self.parameters():
            param.requires_grad = False
    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    @property
    def is_training_layer(self):
        return self.is_training_self

    @is_training_layer.setter # 可以像调用属性那样修改is_training_layer的值
    def is_training_layer(self, other: bool):
        self.is_training_self = other

class SAE(nn.Module):
    def __init__(self, layers_list=None):
        super(SAE,self).__init__()
        self.layers_list = layers_list
        self.initialize()
        self.encoder1 = self.layers_list[0]
        self.encoder2 = self.layers_list[1]
        self.drop = torch.nn.Dropout(0.3)
        self.encoder3 = self.layers_list[2]
        self.encoder4 = self.layers_list[3]

    def initialize(self):
        for layer in self.layers_list:
            layer.is_training_layer = False

    def forward(self,x):
        out = x
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = self.drop(out)
        out = self.encoder3(out)
        out = self.encoder4(out)
        return out
    

class MySAE(nn.Module):
    def __init__(self): 
        super(MySAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            # nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # self.code = self.encoder2.outputs #经过encoder1 之后的输出 是12
        # self.rec_x = self.decoder2.outputs #经过encoder2+decoder 之后的输出 重构的x 32
    def forward(self,x):
        encode = self.encoder(x)
        rec_x = self.decoder(encode)
        return encode, rec_x

class MySAE1(nn.Module):
    def __init__(self): 
        super(MySAE1, self).__init__()
        self.encoder1 = nn.Linear(32, 64)
        self.encoder2 = nn.Linear(64,12)
        self.decoder1 = nn.Linear(12, 64)
        self.decoder2 = nn.Linear(64, 32)

        # self.code = self.encoder2.outputs #经过encoder1 之后的输出 是12
        # self.rec_x = self.decoder2.outputs #经过encoder2+decoder 之后的输出 重构的x 32
    def forward(self,x):
        x = self.encoder1(x)
        x = F.relu(x)
        x = self.encoder2(x)
        encode = F.relu(x)
        x = self.decoder1(encode)
        x = F.relu(x)
        rec_x = self.decoder2(x)
        rec_x = F.relu(rec_x)
        return encode, rec_x
import numpy as np
import torch

data = np.load('/Users/zirs/Desktop/SAE/data/data_122.npy') #(1280, 8064, 12)
fea = np.load('/Users/zirs/Desktop/SAE/data/fea_all.npy') #(1280, 12, 5)
data = torch.from_numpy(data)
fea = torch.from_numpy(fea)

data = data.unsqueeze(3) #(1280, 8064, 12, 1)
fea = fea.unsqueeze(1) #(1280, 1, 12, 5)
data_ = data+fea  #torch.Size([1280, 8064, 12, 5])
data_ = data_.numpy()
np.save('/Users/zirs/Desktop/SAE/data/data_gengraph.npy',data_)
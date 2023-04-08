import numpy as np

y = np.load('/Users/zirs/Desktop/SAE/data/label_val.npy') #(32 , 40, 1)
y =  y.reshape((-1,1)) #(1280, 1)
#将数据和标签组合成字典

labellist= []
for i in range(len(y)):
    label1 = y[i,:]
    if label1<=3:
        label1 = 0
    elif label1<=6:
        label1 = 1
    else:
        label1 = 2
    labellist.append(label1) #(1280, 1)

label = np.array(labellist)

np.save('/Users/zirs/Desktop/SAE/data/label.npy',label)

# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(categories='auto')

# label3 = encoder.fit_transform(label.reshape(-1,1)).toarray()
# print(label3[3,:])
# np.save('/Users/zirs/Desktop/SAE/data/label3.npy', label3)
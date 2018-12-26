#!/usr/bin/env python
# coding: utf-8

# In[17]:


import sys
from PIL import Image
import numpy as np
import csv
import torch
from torch import nn
from torch.autograd import Variable
import cv2
import datetime
import math

LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
batsize=100
num_epochs=2
def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(34)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.extend(onehot)  #every label size is 1*170, according with LETTERSTR
#         labellist.append(onehot)
    return labellist


# In[18]:


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]

    for i in range(num_convs - 1):  # 定义后面的许多层
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.BatchNorm2d(out_channels))
    net.append(nn.MaxPool2d(2, 2))  # 定义池化层
    net.append(nn.Dropout(0.3))
    return nn.Sequential(*net)


# 将模型打印出来看一下结构
block_demo = vgg_block(3, 3,256)
# print(block_demo)


# In[19]:


# 下面我们定义一个函数对这个 vgg block 进行堆叠
def vgg_stack(num_convs, channels):
    net = []
#     print(zip(num_convs, channels))
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


# 作为实例，我们定义一个稍微简单一点的 vgg 结构，其中有 8 个卷积层
vgg_net = vgg_stack((2, 2, 2, 1), ((3, 32), (32, 64), (64, 128), (128, 256)))
# print(vgg_net)


# In[20]:


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(3*12*256, 2560),
            nn.Dropout(0.3),
            nn.Linear(2560, 170)# 34*5=170
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    # 然后我们可以训练我们的模型看看在 cifar10 上的效果
    def data_tf(x):
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5
        x = x.transpose((2, 0, 1))  ## 将 channel 放到第一维，只是 pytorch 要求的输入方式
        x = torch.from_numpy(x)
        return x
    
net = vgg()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MultiLabelSoftMarginLoss()
# print(net)


# In[21]:


print("Reading training data...")
traincsv = open('D:/data/5_imitate_train_set/captcha_train.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(cv2.imread("D:/data/5_imitate_train_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(traincsv)])
train_data = train_data.transpose(0,3,1,2)
traincsv = open('D:/data/5_imitate_train_set/captcha_train.csv', 'r', encoding = 'utf8')    
train_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
# print(train_label[0:2])
print("Shape of train data:", train_data.shape)
# print("Shape of train label:", len(train_label))

print("Reading validation data...")
valicsv = open('D:/data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
vali_data = np.stack([np.array(Image.open("D:/data/5_imitate_vali_set/" + row[0] + ".jpg"))/255.0 for row in csv.reader(valicsv)])
vali_data = vali_data.transpose(0,3,1,2)
valicsv = open('D:/data/5_imitate_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
vali_label = [toonehot(row[1]) for row in csv.reader(valicsv)]
# vali_label = [[] for _ in range(5)]
# for arr in read_label:
#     for index in range(5):
#         vali_label[index].append(arr[index])
# vali_label = [arr for arr in np.asarray(vali_label)]
print("Shape of validation data:", vali_data.shape)
# print("Shape of validation label:", vali_label[0].shape)


# In[15]:


print(vali_data.mean(axis=2))


# In[24]:


def get_acc(output, label):
    correct_num =0
    for i in range(output.size()[0]):
        c0 = np.argmax(output[i, 0:34].data.numpy())
        c1 = np.argmax(output[i, 34:68].data.numpy())
        c2 = np.argmax(output[i, 68:102].data.numpy())
        c3 = np.argmax(output[i, 102:136].data.numpy())
        c4 = np.argmax(output[i, 136:170].data.numpy())
        c = '%s/%s/%s/%s/%s' % (c0, c1, c2, c3,c4)
        l0 = np.argmax(label[i, 0:34].data.numpy())
        l1 = np.argmax(label[i, 34:68].data.numpy())
        l2 = np.argmax(label[i, 68:102].data.numpy())
        l3 = np.argmax(label[i, 102:136].data.numpy())
        l4 = np.argmax(label[i, 136:170].data.numpy())
        l = '%s/%s/%s/%s/%s' % (l0, l1, l2, l3,l4)
        if l==c:
            correct_num += 1        
    return float(correct_num)/len(output)


plt_train_loss = []
plt_train_acc = []

#if torch.cuda.is_available():
#    net = net.cuda()
iters = int(math.ceil(train_data.shape[0]/batsize)) 
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    net = net.train()              
    for i in range(iters):
        im = train_data[i*batsize: (i+1)*batsize]
        im = torch.tensor(im).float()
        label = train_label[i*batsize: (i+1)*batsize]
        label = torch.tensor(label).float()
#        if torch.cuda.is_available():                    
#            im = Variable(im.cuda())
#            label = Variable(label.cuda())
#        else:
#            im = Variable(im)
#            label = Variable(label)
        im = Variable(im)
        label = Variable(label)    
        output = net(im)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = loss.data[0]
    plt_train_loss.append(train_loss.data.numpy())
    
#    if torch.cuda.is_available():
#        train_acc = get_acc(output.cpu(), label.cpu())
#    else:
#        train_acc = get_acc(output, label)
    train_acc = get_acc(output, label)
    plt_train_acc.append(train_acc)
    
    epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                (epoch, train_loss ,train_acc))
    print(epoch_str)
    
        


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, csv, time, random
import torch
import torch.utils.data as Data
import numpy as np
from torch import nn, optim


# In[2]:


input_size = 27
hidden_size = 512
num_layers = 48
batch_size = 225
epochs = 150
num_classes = 44


# In[3]:


base_path = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_path, "dataset")

model_dir = "model"

classes=list(range(1, num_classes + 1))
trainData = list()
testData = list()

for index,name in enumerate(classes):
    label = torch.tensor(index)
    dataset = os.listdir(os.path.join(data_path, str(name)))
    random.shuffle(dataset)
    trainset = dataset[:int(len(dataset) * .8)]
    testset = dataset[int(len(dataset) * .8):]
    for csv_name in trainset:
        print ("Read Train: " + os.path.join(data_path, str(name), csv_name))
        csv_new = list()
        
        csv_path = os.path.join(data_path, str(name), csv_name)
        
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = csv.reader(f)
            rows = list()
            for line in lines:
                row = line[5:8] + line[18:21] + line[31:34] + line[41:59] 
                #row = [line[2], line[6], line[7], line[11], line[12], line[13], line[14], line[15], line[16]] 
                row = np.array(row, dtype='float32')
                rows.append(row)
            trainData.append((torch.tensor(rows).cuda(), label))
                
    for csv_name in testset:
        print ("Read Test: " + os.path.join(data_path, str(name), csv_name))
        csv_new = list()

        csv_path = os.path.join(data_path, str(name), csv_name) #每一个图片的地址

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = csv.reader(f)
            rows = list()
            for line in lines:
                row = line[5:8] + line[18:21] + line[31:34] + line[41:59] 
#                 row = [line[2], line[6], line[7], line[11], line[12], line[13], line[14], line[15], line[16]] 
                row = np.array(row, dtype='float32')
                rows.append(row) 
            testData.append((torch.tensor(rows).cuda(), label))


# In[4]:


class Dataset(Data.Dataset):
 
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        img, label = self.data[index][0], self.data[index][1]
        return (img, label)
 
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return DataLoaderIter(self)


# In[5]:


# DataLoader

trainSet = Dataset(trainData)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
testSet = Dataset(testData)
testloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)


# In[6]:


# Model Configuration


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(    
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
        )

        self.out = nn.Linear(hidden_size, num_classes)    # 输出层

    def forward(self, x):
        
        r_out, (h_n, h_c) = self.rnn(x, None)   

        out = self.out(r_out[:, -1, :])
        return out

rnn = LSTM(input_size, hidden_size, 1, num_classes)
rnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(rnn.parameters(), lr = 0.001)


# In[7]:



# Train Step


for epoch in range(1, epochs + 1):
    total = 0
    correct = 0
    
    
    
    for step, (tx, ty) in enumerate(trainloader):
        
        optimizer.zero_grad()
        tx = tx.cuda()
        ty = ty.cuda()
        py = rnn(tx)
        
        loss = criterion(py, ty)
        loss.backward()
        optimizer.step()  #更新参数
        
        if step == 0:
            _, predicted = torch.max(py, 1)
            total += ty.size(0)
            correct += (predicted == ty).sum()
            acc = float(correct)/float(total)
        
    print('Epoch:', epoch, '|train loss:%.4f' % loss.item(), '|test accuracy:%.4f' % acc)
        
    


# In[34]:


# Test Step

acc_list = list()
loss_list = list()

for step, (tx, ty) in enumerate(testloader):
    tx = tx.cuda()
    ty = ty.cuda()
    py = rnn(tx) 
    loss = criterion(py, ty)
        
    _, predicted = torch.max(py, 1)

    total += ty.size(0)  # 記錄總個數
    correct += (predicted == ty).sum()  # 分配正確的個數
    acc = float(correct)/float(total)
    acc_list.append(acc)
    loss_list.append(loss.item())
    
test_loss = float(np.array(loss_list).sum()/len(loss_list))
test_acc = float(np.array(acc_list).sum()/len(acc_list))
    
print('|train loss:%.4f' % test_loss, '|test accuracy:%.4f' % test_acc)

if not os.path.exists(os.path.join(base_path, model_dir)):
    os.mkdir(os.path.join(base_path, model_dir))

torch.save(rnn.state_dict(), os.path.join(base_path, model_dir, "model.pt"))






# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:58:46 2017

@author: sunrise_forever
"""

import os
os.system('cls')
import gzip, struct
import numpy as np
def _read(image,label):
    minist_dir = 'E:/工作/my work_python/MNIST_data/'
    with gzip.open(minist_dir+label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(minist_dir+image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image,label

def get_data():
    train_img,train_label = _read(
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz')
    test_img,test_label = _read(
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz')
    return [train_img,train_label,test_img,test_label]

train_img,train_label,test_img,test_label = get_data()
#计算目标函数值
def predict(w, x):
    return np.dot(w,x.transpose())
#损失函数值    
def loss_cal(w,dataMat,labelMat,lam):
    loss = 0
    m,n = np.shape(dataMat)
    for i in range(m):
        loss += max([0,1-labelMat[i]*np.dot(w,dataMat[i,:].transpose())])
    loss = loss/m + lam/2*np.linalg.norm(w)**2
    return loss
        
#集中式的梯度下降算法
def batchPegasos(dataSet, labels, lam, T, k):
    loss_store = [];
    m,n = np.shape(dataSet); w = np.zeros(n)*0.1;
    dataIndex = list(range(m))
    for t in range(1, T+1):
        wDelta = np.mat(np.zeros(n)) #reset wDelta
        eta = 1.0/(lam*t)
        np.random.shuffle(dataIndex)
        for j in range(k):#go over training set 
            i = dataIndex[j]
            p = predict(w, dataSet[i,:])        #mapper code
            if (labels[i]*p)[0,0] < 1:                 #mapper code
                wDelta += labels[i]*dataSet[i,:] #accumulate changes  
        w = (1.0 - 1/t)*w + (eta/k)*wDelta       #apply changes at each T        
#        if t%100==0:
        temp_loss = loss_cal(w,dataSet,labels,lam)[0,0]
        loss_store.append(temp_loss)
#        w_store.append(w[0,73])
    return w,loss_store
#分布式的梯度下降算法
def batchPegasos_s(data_Set,label,lam,T,k):
    loss_store_d = []; diff_store = []
    m,n = np.shape(data_Set); w1 = np.zeros(n)*0.1;w2 = np.ones(n)*1.0;w3 = np.ones(n)*(-1.0)
    dataIndex = list(range(m))
    np.random.shuffle(dataIndex)
    dataSet = np.zeros((m,n)); labels = np.zeros(m)
    for i in range(m):
        dataSet[i,:] = data_Set[dataIndex[i],:]
        labels[i] = label[dataIndex[i],:]
    
    dataSet1 = dataSet[:20000,:];dataSet2 = dataSet[20000:40000,:];dataSet3 = dataSet[40000:60000,:]
    labels1 = labels[:20000];labels2 = labels[20000:40000];labels3 = labels[40000:60000]
    num = m//3
    inputs_indexs = [i for i in range(k)]
    last_index = (inputs_indexs[-1]+1)%num
    t = 1
    while t < T+1:
        ###The first node
        wDelta_1 = np.mat(np.zeros(n)) #reset wDelta
        eta = 1.0/(lam*t)
        for j in inputs_indexs:
            p = predict(w1,dataSet1[j,:])
            if (labels1[j]*p) < 1:
                wDelta_1 += float(labels1[j])*dataSet1[j,:]
        w1 = (1.0 - 1/t)*w1 + (eta/k)*wDelta_1
        # The second node
        wDelta_2 = np.mat(np.zeros(n)) #reset wDelta
        eta = 1.0/(lam*t)
        for j in inputs_indexs:
            p = predict(w2,dataSet2[j,:])
            if (labels2[j]*p) < 1:
                wDelta_2 += float(labels2[j])*dataSet2[j,:]
        w2 = (1.0 - 1/t)*w2 + (eta/k)*wDelta_2
        ##The third node
        wDelta_3 = np.mat(np.zeros(n)) #reset wDelta
        eta = 1.0/(lam*t)
        for j in inputs_indexs:
            p = predict(w3,dataSet3[j,:])
            if (labels3[j]*p) < 1:
                wDelta_3 += float(labels3[j])*dataSet3[j,:]
        w3 = (1.0 - 1/t)*w3 + (eta/k)*wDelta_3

        w_1 = 0.8*w1+0.2*w2
        w_2 = 0.6*w2+0.2*w3+0.2*w1
        w_3 = 0.2*w1+0.8*w3        
        ##update for iteration
        w1 = w_1; w2 = w_2; w3 = w_3
        diff_1 = np.linalg.norm(w1-w2);
        diff_2 = np.linalg.norm(w1-w3);
        diff_3 = np.linalg.norm(w2-w3);
        diff = np.max([diff_1,diff_2,diff_3])
#        diff_store.append(np.log10(diff))
        diff_store.append(diff)
        t += 1
        inputs_indexs = []
        for i in range(k):
            inputs_indexs.append((i+last_index)%num)
        last_index = (inputs_indexs[-1]+1)%num
#        if t%100==0:
        loss = loss_cal(w1,data_Set,label,lam)[0,0]
        loss_store_d.append(loss)
    return w1,w2,w3,diff_store,loss_store_d

#图像像素值-向量转化
def img2vector(train_img):
    m = np.shape(train_img)[0]
    returnVect = np.zeros((m,784))
    for i in range(m):
        for j in range(28):
            for k in range(28):
                returnVect[i,j*28+k] = train_img[i,j,k]
    return returnVect
#目标数字
def num_to_train(train_label,num):
    labels = []
    for i in train_label:
        if i == num:
            labels.append(1)
        else:
            labels.append(-1)
    return labels
#规范化
def Normalization(x):
    from sklearn import preprocessing
    Xscaled = preprocessing.scale(x)
    return Xscaled            
dataMat = img2vector(train_img);
labels = num_to_train(train_label,1)
labelMat = np.mat(labels).transpose()
#dataMat = Normalization(dataMat)
w,loss_store = batchPegasos(dataMat,labelMat,10,300,100)
#loss_store = np.log10(loss_store)
w1,w2,w3,diff_store,loss_store_d = batchPegasos_s(dataMat,labelMat,10,300,100)
#loss_store_d = np.log10(loss_store_d)
#print(w_store1,w_store2,w_store3)
def test(dataMat,labelMat,w):
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        prediction = predict(w,dataMat[i,:])[0,0]
        if np.sign(prediction) < 0: temp = -1
        else: temp = 1
        if temp!=np.sign(labelMat[i]): errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))
datMat=img2vector(test_img);
#datMat = Normalization(datMat)
test_labels = num_to_train(test_label,1)
label_Mat = test_labels
test(datMat,label_Mat,w1)
test(datMat,label_Mat,w2)
test(datMat,label_Mat,w3)
test(datMat,label_Mat,w)
print(diff_store[-1])
x1 = range(300)
x2 = range(300)
import matplotlib.pyplot as plt
ax1 = plt.figure().add_subplot(111)
#diff_1 = []
#for i in range(500):
#    diff_1.append(diff_store[i*2])

    
ax1.plot(x2,diff_store,'b-')
#ax1.plot(x,w_store1,'b',label='node1')
#ax1.plot(x,w_store2,'r',label='node2')
#ax1.plot(x,w_store3,'g',label='node3')
plt.legend()
plt.xlabel('Iterations');
plt.ylabel('D(t)')
plt.yscale('log')
#plt.title('The consensus rate of parameter estimation by DSSGD')
plt.savefig('DSSGD consensus',format='eps',dpi=1000)
plt.show()
ax2 = plt.figure().add_subplot(111)
ax2.plot(x1,loss_store,'r:',label='Pegasos')
ax2.plot(x1,loss_store_d,'b-',label='DSSGD')
plt.legend()
plt.xlabel('Iterations');plt.ylabel('F(w)')
plt.yscale('log')
#plt.title('The decreasing curve of objective function')
plt.savefig('DSSGD objective',format='eps',dpi=1000)
plt.show()

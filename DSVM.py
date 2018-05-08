# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 21:22:47 2017

@author: sunrise_forever
"""
import time
from numpy import *
import pandas as pd
def load_spam_data(filename):
    fr = open(filename)
    array = fr.readlines()
    row_n = len(array)
    col_n = len(array[0].strip().split(','))
    dataMat = ones((row_n,col_n))
    labelMat = zeros(row_n)
    index = 0
    for line in array:
        linelist = line.strip().split(',')
        temp = linelist[:-1]
        dataMat[index,0:col_n-1] = [float(x) for x in temp]
        if int(linelist[-1])==0:
            labelMat[index] = -1
        else:
            labelMat[index] = 1
        index +=1
    dataMat = Normalization(dataMat)
    return dataMat,labelMat


def Normalization(x):
    from sklearn import preprocessing
    Xscaled = preprocessing.scale(x)
    return Xscaled
    
#def seqPegasos(dataSet, labels, lam, T):
#    m,n = shape(dataSet); w = zeros(n)*0.0
#    for t in range(1, T+1):
#        i = random.randint(m)
#        eta = 1.0/(lam*t)
#        p = predict(w, dataSet[i,:])
#        if labels[i]*p < 1:
#            w = (1.0 - 1/t)*w + eta*labels[i]*dataSet[i,:]
#        else:
#            w = (1.0 - 1/t)*w
#        print (w)
#    return w
        
def predict(w, x):
    return dot(w,x.transpose())
    
def loss_cal(w,dataMat,labelMat,lam):
    loss = 0
    m,n = shape(dataMat)
    for i in range(m):
        loss += max([0,1-labelMat[i]*dot(w,dataMat[i,:].transpose())])
    loss = loss/m + lam/2*linalg.norm(w)**2
    return loss
        

def batchPegasos_D(dataSet, labels, lam, T, k):
    loss_list = []; w_store1 = []; w_store2 = []; w_store3 = []
    m,n = shape(dataSet); w1 = zeros(n)*0.0; w2 = ones(n)*1.0;w3 = ones(n)*(-1.0);
    dataIndex = list(range(m))
    for t in range(1, T+1):
        wDelta_1 = mat(zeros(n)); wDelta_2 = mat(zeros(n));wDelta_3 = mat(zeros(n));#reset wDelta
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        ###adaption
        for j in range(k):#go over training set 
            i = dataIndex[j]
            p = predict(w1, dataSet[i,:])        #mapper code
            if (labels[i]*p).all() < 1:                 #mapper code
                wDelta_1 += labels[i]*dataSet[i,:].transpose() #accumulate changes  
        w1 = (1.0 - 1/t)*w1 + (eta/k)*wDelta_1       #apply changes at each T
        w_store1.append(w1[0,0])
        for j in range(k,2*k):
            i = dataIndex[j]
            p = predict(w2, dataSet[i,:])        #mapper code
            if (labels[i]*p).all() < 1:                 #mapper code
                wDelta_2 += labels[i]*dataSet[i,:].transpose() #accumulate changes  
        w2 = (1.0 - 1/t)*w2 + (eta/k)*wDelta_2
        w_store2.append(w2[0,0])
        for j in range(2*k,3*k):
            i = dataIndex[j]
            p = predict(w3, dataSet[i,:])        #mapper code
            if (labels[i]*p).all() < 1:                 #mapper code
                wDelta_3 += labels[i]*dataSet[i,:].transpose() #accumulate changes  
        w3 = (1.0 - 1/t)*w3 + (eta/k)*wDelta_3
        w_store3.append(w3[0,0])
        ###combination
        w_1 = 0.8*w1+0.2*w2
        w_2 = 0.6*w2+0.4*w3
        w_3 = 0.2*w1+0.2*w2+0.6*w3
        ###update for iteration
        w1 = w_1; w2 = w_2; w3 = w_3
        temp_loss = loss_cal(w1,dataSet,labels,lam)[0,0]
        loss_list.append(temp_loss)
    return w1,w2,w3,w_store1,w_store2,w_store3,loss_list
    
def batchPegasos(dataSet, labels, lam, T, k):
    loss_list = []
    w_store = []
    m,n = shape(dataSet); w = zeros(n)*0.0; 
    dataIndex = list(range(m))
    for t in range(1, T+1):
        wDelta = mat(zeros(n)) #reset wDelta
        eta = 1.0/(lam*t)
        random.shuffle(dataIndex)
        for j in range(k):#go over training set 
            i = dataIndex[j]
            p = predict(w, dataSet[i,:])        #mapper code
            if (labels[i]*p).all() < 1:                 #mapper code
                wDelta += labels[i]*dataSet[i,:].transpose() #accumulate changes  
        w = (1.0 - 1/t)*w + (eta/k)*wDelta       #apply changes at each T
        w_store.append(w[0,0])
        temp_loss = loss_cal(w,dataSet,labels,lam)[0,0]
#        print(temp_loss)
        loss_list.append(temp_loss)
    return w,w_store,loss_list
    
def test(dataMat,labelMat,w):
    n = len(labelMat)
    count = 0 
    for i in range(n):
        temp = predict(w,dataMat[i,:])
        if temp > 0 :
            y=1
        else:
            y=-1
        if y == labelMat[i]:
            count += 1
    accuracy = float(count/n)
    print('The accuracy on training set is %f.' %accuracy)
    return accuracy

        
        
t1 = time.time()    
dataMat,labelList = load_spam_data('E:/工作/my work_python/spambase.data')
count = 0
for i in labelList:
    if i == -1:
        count += 1 
print('The rate of negative sample is %f.'%float(count/len(labelList)))

        
    
t2 = time.time()
print('It costs %d seconds to preprocess the spamdata.'%int(t2-t1))
t3 = time.time()
#accuracy = 0.0
#for lam in [0.001,0.01,0.05,0.1,0.5,1,5,10,50,100]:
#    finalWs,w_store,loss_list = batchPegasos(dataMat, labelList,lam,100,300)
#    temp_acc = test(dataMat,labelList,finalWs)
#    if temp_acc > accuracy:
#        accuracy = temp_acc
#        w = finalWs;lam_opt = lam   
#print(lam_opt)
finalWs,w_store,loss_list = batchPegasos(dataMat, labelList,0.1,1000,300)
t4 = time.time()
finalW_1,finalW_2,finalW_3,w_store1,w_store2,w_store3,loss_list_D = batchPegasos_D(dataMat, labelList, 0.1, 1000, 300)
t5 = time.time()
print('Pegasos and DSVM cost %d  and %d seconds respectively.'%(int(t4-t3),int(t5-t4)))
print(linalg.norm(finalWs),linalg.norm(finalW_1))
#test(dataMat,labelList,w)
test(dataMat,labelList,finalW_1)
test(dataMat,labelList,finalW_2)
test(dataMat,labelList,finalW_3)
x = range(1000)
import matplotlib.pyplot as plt
plt.plot(x,w_store1,'b',x,w_store2,'r',x,w_store3,'g',x,w_store,'y')
plt.show()
plt.plot(x,loss_list,'r',x,loss_list_D,'b')
plt.show()
#print(finalW_1)
#print(loss_list[-1])


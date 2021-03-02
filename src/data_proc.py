import cv2
import numpy as np
import random
import datetime
import os


def load_data(path): 
    print(datetime.datetime.now(),'loading') 
    dataset={}       
    count   = 0
    for dir in os.listdir(path):
        print(datetime.datetime.now(),dir)
        dataset[dir]=np.array([])
        count      += 1
        for t in os.listdir(os.path.join(path,dir)):
            img    = cv2.imread(os.path.join(path,dir,t))
            img    = img[:,:,0]
            dataset[dir]=np.append(dataset[dir],img)
        (h,w) = img.shape
        dataset[dir]=dataset[dir].reshape((-1,h,w,1))
    return dataset,(h,w,1)


def data_split(dataset,input_shape):
    count=0   
    _dataset = np.array([])  
    labels  = np.array([]) 
    num_classes=dataset.__len__()
    for key in dataset:
        label   = np.zeros(num_classes)
        label[count]=1
        count+=1
        totle=dataset[key].shape[0]
        shuffle=random.sample([i for i in range(totle)],totle)
        for i in range(totle):
            _dataset= np.append(_dataset,dataset[key][shuffle[i]])
            labels  = np.append(labels,label)
    (h,w,c)=input_shape
    _dataset=_dataset.reshape((-1,h,w,c))
    labels  = labels.reshape((-1,num_classes))
    return _dataset,labels


def data_norm(dataset,labels):
    _dataset=[]
    _labels=[]
    for i in range(len(dataset)):
        pic=np.float32(dataset[i])
        if np.max(pic) != np.min(pic) and np.std(pic) != 0:
            pic     = (pic-np.min(pic))/(np.max(pic)-np.min(pic))
            pic     = (pic-np.mean(pic))/np.std(pic)
            _dataset.append(pic)
            _labels.append(labels[i])
    h,w=pic.shape[0],pic.shape[1]
    _dataset=(np.array(_dataset)).reshape((-1,w,h,1))
    _labels=(np.array(_labels)).reshape((-1,labels.shape[1]))
    return _dataset,_labels


def shuffle_data(dataset_labels):
    (dataset,labels)=dataset_labels
    shuffle = [i for i in range(len(dataset))] 
    shuffle = random.sample(shuffle, len(shuffle))
    t1      = dataset[shuffle[0]]
    t2      = labels[shuffle[0]]
    for i in range(len(shuffle)-1):
        dataset[shuffle[i]] = dataset[shuffle[i+1]]
        labels[shuffle[i]]  = labels[shuffle[i+1]]
    dataset[len(shuffle)-1] = t1
    labels[len(shuffle)-1]  = t2
    return dataset, labels

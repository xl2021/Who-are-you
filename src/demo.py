from keras.models import Sequential, load_model
import cv2
import os
import numpy as np


model_name='49_model.h5'
path='./ce5'
dst = './demo/demo.mp4'
fps = 25
size = (848, 478)
font = cv2.FONT_HERSHEY_SIMPLEX
cnt=0
all={}


def get_model(model_name=model_name):
    model=Sequential()
    model=load_model(model_name)
    return model


def get_prob(img,org,model):
    img=img[org[1]:org[1]+68, org[0]:org[0]+68, 0:1]
    if np.max(img) != np.min(img) and np.std(img) != 0:
        img     = (img-np.min(img))/(np.max(img)-np.min(img))
        img     = (img-np.mean(img))/np.std(img)
    img=img.reshape((1,68,68,1))
    classes=model.predict(img)
    prob=classes[0].argsort()[::-1]
    return prob


def get_find(im,i,j):
    if j==34:
        for m in range(68):
            for n in range(34):
                if im[i-34+m][j-1-n]>200:
                    return True
    if j==813:        #848-34-1
        for m in range(68):
            for n in range(34):
                if im[i-34+m][j+1+n]>200:
                    return True
    return im[i][j]>200


def get_rec(img):
    find=False
    im=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x,y=278+34,0+34
    max=0
    for i in range(312,444):       #278+34, 478-34
        for j in range(34,814):        #848-34
            if not get_find(im,i,j):
                continue
            find=True
            sum=0
            for m in range(68):
                for n in range(68):
                    if im[i-34+m][j-34+n]>200:
                        sum+=1
            if sum>max:
                x=np.array([i])
                y=np.array([j])
                max=sum
            elif sum==max:
                x=np.append(x,i)
                y=np.append(y,j)
    x,y=int(np.mean(x)),int(np.mean(y))
    return y,x,find


def draw_rec(img):
    x,y,find=get_rec(img)
    org = (x-34, y-34)
    end = (x+34, y+34)
    cv2.rectangle(img, org, end, (0,0,255),1)
    return img, org, find


def draw_text(img,prob,cnt,all):
    ids={0: 'cat', 1: 'desert_fox', 2: 'dog', 3: 'manul', 4: 'marmot', 5: 'marten', 
        6: 'mongolia_rabbit', 7: 'rabbit', 8: 'red_fox', 9: 'weasel', 10: 'wolf'}
    text_point_1 = (25, 100)
    text='For current frame:'
    text_size, _ = cv2.getTextSize(text, font, 0.5, 1)
    cv2.putText(img, text, text_point_1, font, 0.5, (255,255,255), 1)
    for i in range(len(prob)):
        text='Top '+str(i+1)+':'+ids[prob[i]]
        text_point_1=(text_point_1[0],text_point_1[1]+int(text_size[1]*1.2))
        cv2.putText(img, text, text_point_1, font, 0.5, (0,255,0), 1)
        for j in range(i+1,12):
            if j not in all:
                all[j]={}
            if ids[prob[i]] not in all[j]:
                all[j][ids[prob[i]]]=1
            else:
                all[j][ids[prob[i]]]+=1

    text_point_2 = (550, 100)
    text='For all frames up to now:'
    cv2.putText(img, text, text_point_2, font, 0.5, (255,255,255), 1)
    for i in range(11):
        std_all=sorted(all[i+1].items(), key=lambda item:item[1], reverse=True)
        text='Top '+str(i+1)+':'+std_all[0][0]+'('+str(std_all[0][1])+'/'+str(cnt)+')'
        text_point_2=(text_point_2[0],text_point_2[1]+int(text_size[1]*1.2))
        cv2.putText(img, text, text_point_2, font, 0.5, (0,255,0), 1)
    return img


if __name__ == '__main__':
    model=get_model()
    videowriter = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)
    for pic in os.listdir(path):
        img=cv2.imread(os.path.join(path,pic))
        img,org,find=draw_rec(img)
        if find:
            cnt+=1
            prob=get_prob(img,org,model)
            img=draw_text(img,prob,cnt,all)
        videowriter.write(img)
        cv2.imwrite('./demo/'+pic,img)

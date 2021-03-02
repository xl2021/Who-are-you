from data_proc import *
from my_model import *
from keras.optimizers import Adam
import datetime


train_dir   = 'train_set'
val_dir     = 'val_set'
pretrain    = 'model.h5'       


def train(dataset, val_dataset, input_shape, pretrain, optimizer=Adam(lr=0.001), batch_size=32):
    start=0
    histories={}

    use_pretrain=''
    sv_name = ''
    while(use_pretrain!='y' and use_pretrain!='n'):
        use_pretrain=input('Use pre-model? [y/n] : ')
    if use_pretrain == 'n':
        model=get_model(input_shape=input_shape, num_classes=dataset.__len__(), optimizer=optimizer)
        sv_name='initial_model.h5'
        model.save(sv_name)
        print('Model created')
    else:
        while(not os.path.exists(sv_name)):
            sv_name=input('Model name: ')
        model=get_model(pretrain=sv_name,optimizer=optimizer)
        print("Model '"+sv_name+"' got")

    raw_dataset,raw_labels=data_split(dataset,input_shape)
    train_dataset, train_labels=shuffle_data(data_norm(raw_dataset,raw_labels))
    print('Dataset shuffled')
    
    val_dataset,val_labels=data_split(val_dataset,input_shape)
    val_dataset,val_labels=data_norm(val_dataset,val_labels)

    next_steps=int(input('Epoches: '))
    
    while(True):
        if start>=next_steps:
            cont=''
            while(cont!='y' and cont!='n'):
                cont=input('Continue training? [y/n] : ')
            if cont=='n':
                break
            
            model=get_model(pretrain=sv_name,optimizer=optimizer)
            train_dataset, train_labels=data_norm(raw_dataset, raw_labels)
            
            apply_shuffle=''
            while(apply_shuffle!='y' and apply_shuffle!='n'):
                apply_shuffle=input('Shuffle data? [y/n] : ')
            if apply_shuffle=='y':
                train_dataset, train_labels=shuffle_data((train_dataset,train_labels))
                print('Dataset shuffled')
            
            next_steps+=int(input('Next epoches: '))
        else:
            model=get_model(pretrain=sv_name,optimizer=optimizer)
        
        if start>=next_steps:
            continue
        
        print(datetime.datetime.now(),'training --',str(start),'epoches:')
        history = model.fit(train_dataset, train_labels, batch_size=batch_size, epochs=1, 
                        validation_data=(val_dataset, val_labels))
        sv_name = str(start) + '_' + pretrain
        model.save(sv_name)
        print("model '"+sv_name+"' saved")
        
        for key in history.history:
            if key not in histories:
                histories[key]=history.history[key]
            else:
                for v in history.history[key]:
                    histories[key].append(v)
        
        start+=1
        
    return histories


if __name__ == '__main__': 
    dataset, (height,width,channel) = load_data(train_dir)
    #val_dataset & val_labels need to apply function 'data_norm()' before using
    val_dataset, (height,width,channel) = load_data(val_dir)
    histories=train(dataset, val_dataset, optimizer=Adam(lr=0.001), input_shape=(width,height,channel),
            pretrain=pretrain)
    #print(histories)
    
    print(datetime.datetime.now(),'the end')

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model

def get_model(input_shape=None, num_classes=11, pretrain=None, optimizer=Adam(lr=0.001)):
    if pretrain == None:
        input=Input(shape=input_shape)
        x=Conv2D(4,(5,5),activation='relu',name='layer_0')(input)
        x=MaxPooling2D(pool_size=(2,2))(x)
        x=Conv2D(16,(5,5),activation='relu',name='layer_1')(x)
        x=MaxPooling2D(pool_size=(2,2))(x)
        x=Conv2D(64,(5,5),activation='relu',name='layer_2')(x)
        x=MaxPooling2D(pool_size=(2,2))(x)
        x=Flatten()(x)
        x=Dense(256,activation='relu',name='layer_3')(x)
        x=Dropout(0.5)(x)
        x=Dense(64,activation='relu',name='layer_4')(x)
        output=Dense(num_classes, activation='softmax',name='layer_5')(x)
        model=Model(input, output)
        model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    else:
        model   = Sequential()
        model   = load_model(pretrain)
    #print(model.summary())
    return model


if __name__ == '__main__':
    model=get_model(input_shape=(68,68,1))
    plot_model(model, to_file='model.png', show_shapes=True)

'''
https://drive.google.com/file/d/0B3zSzVUgsG-FNTliNFl2a2t5dzQ/view?usp=sharing
'''

import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers.convolutional import Convolution2D, Convolution1D, AveragePooling2D, AveragePooling1D, MaxPooling1D, MaxPooling2D, ZeroPadding1D, ZeroPadding2D
from keras.optimizers import *

def load_dump(dump_file):
    fp = open(dump_file, 'rb')
    if not fp:
        print('Fail to open the dump file: %s' % dump_file)
        return None
    dump = pickle.load(fp)
    fp.close()
    return dump

def defnet1(nb_filter_base = 64, nb_mid_blk = 1):
    '''A VGG-like net
    '''
    input_dim = 128
    prefix = ''
    act_type = 'sigmoid'
    act_type = 'tanh'
    act_type = 'relu'
    init_type = 'uniform'
    init_type = 'glorot_uniform'
    init_type = 'he_normal'
    init_type = 'he_uniform'

    model = Sequential()
    model.add(Convolution2D(nb_filter_base, 2, 3, init=init_type, activation=act_type, input_shape=(1, 2, input_dim), border_mode='valid', name=prefix+'conv_input'))
    model.add(Dropout(0.5))

    for k in range(nb_mid_blk):
        nb_filter = nb_filter_base * (2**k)
        model.add(Convolution2D(nb_filter, 1, 3, init=init_type, activation=act_type, border_mode='same', name=prefix+'conv{}_1'.format(k+1)))
        model.add(Convolution2D(nb_filter, 1, 3, init=init_type, activation=act_type, border_mode='same', name=prefix+'conv{}_2'.format(k+1)))
        model.add(AveragePooling2D((1,2), strides=(1,2), name=prefix+'pool{}'.format(k+1)))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax', name=prefix+'softmax'))

    optimizer = Adam()
    optimizer = Adadelta()
    optimizer = SGD(lr=1e-4, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    return model


def train():
    #load data
    train_data = load_dump('train_data.pkl')
    train_target = load_dump('train_target.pkl')
    validate_data = load_dump('validate_data.pkl')
    validate_target = load_dump('validate_target.pkl')

    model = defnet1(nb_filter_base=64, nb_mid_blk=1)

    result = model.fit(train_data, train_target,
                       nb_epoch=10, batch_size=128,
                       validation_data=(validate_data, validate_target),
                       class_weight={0: 1, 1:10, 2:20},
                       verbose=2)




if __name__ == '__main__':
    train()

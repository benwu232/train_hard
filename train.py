'''
https://drive.google.com/file/d/0B3zSzVUgsG-FNTliNFl2a2t5dzQ/view?usp=sharing
'''

import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers.convolutional import Convolution2D, Convolution1D, AveragePooling2D, AveragePooling1D, MaxPooling1D, MaxPooling2D, ZeroPadding1D, ZeroPadding2D
from keras.optimizers import *
from keras.callbacks import Callback

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
    act_type = 'tanh'
    act_type = 'linear'
    act_type = 'relu'
    act_type = 'sigmoid'
    act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

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

    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax', name=prefix+'softmax'))

    optimizer = Adam()
    optimizer = Adadelta()
    optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    return model



class LearningRateMonitor(Callback):
    '''Learning rate monitor, monitor the learning rate and change it conditionally

    # Arguments
        epoch_num: monitor range
        shrinkage: shrinkage of learning rate
    '''
    def __init__(self, epoch_num=10, shrinkage=3.0, min_lr=1e-7):
        self.epoch = []
        self.history = {}
        self.epoch_num = epoch_num
        self.shrinkage = shrinkage
        self.min_lr = min_lr
        self.pre_sum = 0.0
        self.cur_sum = 0.0

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}
        self.pause_cnt = 0

    def on_epoch_end(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'

        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        #if epoch < self.epoch_num * 2:
        #    return

        lr = float(K.get_value(self.model.optimizer.lr))
        if lr < self.min_lr:
            self.model.stop_training = True
            return

        self.pause_cnt += 1
        if self.pause_cnt < self.epoch_num:
            return

        type_name = 'val_acc'
        pre_sum = sum(self.history[type_name][-self.epoch_num*2:-self.epoch_num])
        cur_sum = sum(self.history[type_name][-self.epoch_num:])
        if cur_sum < pre_sum:
            lr = float(K.get_value(self.model.optimizer.lr))

            assert type(lr) == float, 'The output of the "schedule" function should be float.'
            new_lr = float(lr / self.shrinkage)
            K.set_value(self.model.optimizer.lr, new_lr)
            print('************* Change learning rate to: %f *****************' % new_lr)
            self.pause_cnt = 0

def train():
    #load data
    train_data = load_dump('train_data.pkl')
    train_target = load_dump('train_target.pkl')
    validate_data = load_dump('validate_data.pkl')
    validate_target = load_dump('validate_target.pkl')

    model = defnet1(nb_filter_base=64, nb_mid_blk=1)
    lr_monitor = LearningRateMonitor(epoch_num=10, shrinkage=2.0, min_lr=1e-5)

    result = model.fit(train_data, train_target,
                       nb_epoch=1000, batch_size=128,
                       validation_data=(validate_data, validate_target),
                       class_weight={0:1, 1:20, 2:20},
                       callbacks=[lr_monitor],
                       verbose=2)

    model.save('final_model.json')
    model.save_weights('final_model.h5')


if __name__ == '__main__':
    train()

# train_hard

I'm new to deep learning. I'm now use Keras and Tensorflow as backend.

I use some real stock market data as my exercise. 

You can download the data from https://drive.google.com/file/d/0B3zSzVUgsG-FNTliNFl2a2t5dzQ/view?usp=sharing

It is more than 600M bytes. 

The training data is very unblanced: 

0: No action, 90%

1: Buy, 5.5%

2: Sell, 4.5%

The shape of the data is like this (412297, 1, 2, 128)

First one(4122297) is number of samples.

The second one(1), is for using convolution2d (I'd like to think about the data as a image).

(2, 128) means close price and volume as two rows, which as 128 values respectively. It is actually two sequence with 128 days' close price and volume. The equation new_v = log(1+v) is applied to the volume row to make it smaller and smoother. It is said this can help train the neural network, but I didn't see that:(. The common normalization(subtracted by mean and divided by standard division) algorithm is applied in each row to make them zero mean and symmetric.

After downloding the file, please extract it in the train_hard directory and run the train.py.

Here is my problem:
When training, I always got nan in loss in first several epochs, even I set lr = 1e-4. 
I don't know why. Is there any black magic in training deep neural network? 
Please give me some advices. Since I am new in this area, any advice is welcome.

Thanks in advance!

Here is the training process.

Train on 412297 samples, validate on 333981 samples

Epoch 1/10

126s - loss: nan - acc: 0.6683 - val_loss: 0.0000e+00 - val_acc: 0.9018

Epoch 2/10

70s - loss: 0.0000e+00 - acc: 0.8968 - val_loss: 0.0000e+00 - val_acc: 0.9018

Epoch 3/10

69s - loss: 0.0000e+00 - acc: 0.8968 - val_loss: 0.0000e+00 - val_acc: 0.9018

# train_hard

I'm new to deep learning. I'm now use Keras and Tensorflow as backend.

I use some real stock market data as my exercise. 

You can download the data from https://drive.google.com/file/d/0B3zSzVUgsG-FNTliNFl2a2t5dzQ/view?usp=sharing

It is more than 600M bytes. 

The training data is very unblanced: 

0: No action, 90%

1: Buy, 5.5%

2: Sell, 4.5%

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

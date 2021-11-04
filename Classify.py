#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from core import generate_sample_group
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np
import random


# In[2]:


def generate_num_nonzero(num_samples, lognorm_mean=3, lognorm_sigm=1):
    # Select number of nonzero terms in spectroscopy sample
    return np.random.lognormal(lognorm_mean, lognorm_sigm, num_samples).astype(int)

def get_locs(num_nonzero, low=0, high=300):
    # Generate the location of the num_nonzero components
    counts = np.random.uniform(0, 1, num_nonzero)
    intensity_locs = counts / counts.sum()
    mass_locs = np.random.uniform(low, high, num_nonzero)
    return intensity_locs, mass_locs

def generate_mass_samples(num_nonzero, num_samples,
                          intensity_locs=[], mass_locs=[],
                          mass_std=0.001, intensity_spread_factor=1.0, intensity_std=0.01):
    samples = []
    for i in range(num_nonzero):
        mass_samples = np.random.normal(mass_locs[i], mass_std, num_samples)
        intensity_samples = np.random.normal(intensity_locs[i], intensity_std, num_samples)
        samples.append(np.column_stack((mass_samples, intensity_samples)))
    
    samples = np.asarray(list(zip(*samples)))
    # ToDo: Improve. This is a weak fix.
    samples[samples < 0] = 0
    return samples

class Generator:
    """ Simulate spectrogram-like samples and vend these samples so that
    they are dispensed in groups of same sequence length.
    """
    def __init__(self, num_groups=10, num_samples_per_group=50):
        self.num_groups = num_groups
        self.num_samples_per_group = num_samples_per_group
        self.data_setup = {}
        self.numNonzero_to_groupIDs = {}
        # Save means of mass and intensity
        for group_id in range(num_groups):
            num_nonzero = generate_num_nonzero(1)[0]
            intensity_locs, mass_locs = get_locs(num_nonzero)
            self.data_setup[group_id] = {'num_nonzero': num_nonzero,
                                         'intensity_locs': intensity_locs,
                                         'mass_locs': mass_locs}
            if num_nonzero not in self.numNonzero_to_groupIDs.keys():
                self.numNonzero_to_groupIDs[num_nonzero] = []
            self.numNonzero_to_groupIDs[num_nonzero].append(group_id)

    def generate_data(self, **generate_sample_kwargs):
        # TODO: improve- make the train_generator a class that way we can iterate through all samples
        # instead of choosing randomly
        num_nonzero = np.random.choice(list(self.numNonzero_to_groupIDs.keys()))
        group_ids_iter = self.numNonzero_to_groupIDs[num_nonzero]
        X = []
        y = []
        for group_id in group_ids_iter:
            settings = self.data_setup[group_id]
            samples = generate_mass_samples(num_nonzero, self.num_samples_per_group,
                                            intensity_locs=settings['intensity_locs'],
                                            mass_locs=settings['mass_locs'],
                                            **generate_sample_kwargs)
            X.extend(samples)
            y.extend([group_id]*self.num_samples_per_group)
        X = np.asarray(X)
        y = to_categorical(y, num_classes=self.num_groups)

        # Shuffle
        Xy = list(zip(X, y))
        random.shuffle(Xy)
        X, y = zip(*Xy)
        X=np.asarray(X)
        y=np.asarray(y)
        return X, y
    
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(None, 2)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(10, activation='sigmoid'))

print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

testX, testY, testLength = [], [], []
def train_generator():
    global testX, testY, testLength
    gen = Generator()
    while True:
        sequence_length = generate_num_nonzero(1, lognorm_mean=3, lognorm_sigm=1)[0]

        X, y = gen.generate_data()

        # Split into train-test splits
        train_pct = 0.9
        train_split = int(train_pct * len(y))

        trainX, trainy = X[:train_split], y[:train_split]
        testX.extend(X[train_split:])
        testY.extend(y[train_split:])
        testLength.extend([sequence_length]*len(y[train_split:]))
        print('\n',trainX.shape, trainy.shape)
        yield trainX, trainy


model.fit_generator(train_generator(), steps_per_epoch=10, epochs=10, verbose=1)



# In[ ]:

# Evaluate on a each unique sequence length

testLength = np.array(testLength)
for sequence_len in np.unique(testLength):
    mask = testLength == sequence_len
    X = [testX[i] for i in mask]
    y = [testY[i] for i in mask]
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    print(y.shape)
    score = model.evaluate(X, y, verbose=0)
    print('Sequence length:', sequence_len)
    print('\tTest loss:', score[0])
    print('\tTest accuracy:', score[1])



"""
Below is the training and test output
"""

##Epoch 1/10
##
## (45, 37, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 31s - loss: 2.1640 - accuracy: 0.0000e+00
## (45, 18, 2) (45, 10)
##
## (45, 21, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 1.9024 - accuracy: 0.3333
## (45, 43, 2) (45, 10)
##
## (45, 37, 2) (45, 10)
## 5/10 [==============>...............] - ETA: 0s - loss: 2.3884 - accuracy: 0.2000
## (45, 75, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
## 7/10 [====================>.........] - ETA: 0s - loss: 2.3124 - accuracy: 0.2857
## (45, 8, 2) (45, 10)
##
## (45, 43, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
##10/10 [==============================] - 4s 34ms/step - loss: 2.4567 - accuracy: 0.2000
##Epoch 2/10
##
## (45, 23, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 2.6623 - accuracy: 0.0000e+00
## (45, 43, 2) (45, 10)
##
## (45, 43, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 2.7552 - accuracy: 0.0000e+00
## (45, 18, 2) (45, 10)
##
## (45, 18, 2) (45, 10)
## 5/10 [==============>...............] - ETA: 0s - loss: 2.5211 - accuracy: 0.0000e+00
## (45, 17, 2) (45, 10)
##
## (45, 7, 2) (45, 10)
##
## (45, 37, 2) (45, 10)
##
## (45, 7, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 2.3920 - accuracy: 0.1111
## (45, 7, 2) (45, 10)
##10/10 [==============================] - 0s 24ms/step - loss: 2.4441 - accuracy: 0.1000
##Epoch 3/10
##
## (45, 37, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 2.8593 - accuracy: 0.0000e+00
## (45, 75, 2) (45, 10)
##
## (45, 8, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 2.1715 - accuracy: 0.3333
## (45, 23, 2) (45, 10)
##
## (45, 18, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
##
## (45, 7, 2) (45, 10)
## 7/10 [====================>.........] - ETA: 0s - loss: 2.2756 - accuracy: 0.1429
## (45, 8, 2) (45, 10)
##
## (45, 43, 2) (45, 10)
##
## (45, 26, 2) (45, 10)
##10/10 [==============================] - 0s 27ms/step - loss: 2.2718 - accuracy: 0.1000
##Epoch 4/10
##
## (45, 23, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 3.3195 - accuracy: 0.0000e+00
## (45, 18, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
##
## (45, 37, 2) (45, 10)
## 4/10 [===========>..................] - ETA: 0s - loss: 2.3922 - accuracy: 0.2500
## (45, 37, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
## 6/10 [=================>............] - ETA: 0s - loss: 2.2151 - accuracy: 0.5000
## (45, 18, 2) (45, 10)
##
## (45, 37, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 2.1443 - accuracy: 0.4444
## (45, 75, 2) (45, 10)
##10/10 [==============================] - 0s 32ms/step - loss: 2.1585 - accuracy: 0.4000
##Epoch 5/10
##
## (45, 17, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 1.5774 - accuracy: 1.0000
## (45, 18, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
## 4/10 [===========>..................] - ETA: 0s - loss: 1.9489 - accuracy: 0.2500
## (45, 8, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
##
## (45, 8, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
## 8/10 [=======================>......] - ETA: 0s - loss: 2.0923 - accuracy: 0.1250
## (45, 23, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
##10/10 [==============================] - 0s 18ms/step - loss: 2.0494 - accuracy: 0.2000
##Epoch 6/10
##
## (45, 43, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 1.8351 - accuracy: 1.0000
## (45, 26, 2) (45, 10)
##
## (45, 37, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 2.3953 - accuracy: 0.3333
## (45, 75, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
## 5/10 [==============>...............] - ETA: 0s - loss: 2.1121 - accuracy: 0.4000
## (45, 17, 2) (45, 10)
## 6/10 [=================>............] - ETA: 0s - loss: 1.9911 - accuracy: 0.5000
## (45, 26, 2) (45, 10)
##
## (45, 18, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 2.0712 - accuracy: 0.4444
## (45, 18, 2) (45, 10)
##10/10 [==============================] - 0s 44ms/step - loss: 1.9611 - accuracy: 0.5000
##Epoch 7/10
##
## (45, 26, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 2.1627 - accuracy: 0.0000e+00
## (45, 17, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
## 4/10 [===========>..................] - ETA: 0s - loss: 1.9574 - accuracy: 0.5000
## (45, 17, 2) (45, 10)
##
## (45, 43, 2) (45, 10)
##
## (45, 43, 2) (45, 10)
## 7/10 [====================>.........] - ETA: 0s - loss: 1.8859 - accuracy: 0.5714
## (45, 8, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 1.9816 - accuracy: 0.4444
## (45, 26, 2) (45, 10)
##10/10 [==============================] - 0s 24ms/step - loss: 1.9348 - accuracy: 0.5000
##Epoch 8/10
##
## (45, 17, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 2.0383 - accuracy: 0.0000e+00
## (45, 37, 2) (45, 10)
##
## (45, 43, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 1.9165 - accuracy: 0.3333
## (45, 17, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
## 5/10 [==============>...............] - ETA: 0s - loss: 1.7739 - accuracy: 0.4000
## (45, 7, 2) (45, 10)
##
## (45, 17, 2) (45, 10)
##
## (45, 26, 2) (45, 10)
##
## (45, 18, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 1.7276 - accuracy: 0.4444
## (45, 17, 2) (45, 10)
##10/10 [==============================] - 0s 23ms/step - loss: 1.7931 - accuracy: 0.4000
##Epoch 9/10
##
## (45, 23, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 0.9200 - accuracy: 1.0000
## (45, 37, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 1.6917 - accuracy: 0.3333
## (45, 43, 2) (45, 10)
## 4/10 [===========>..................] - ETA: 0s - loss: 1.4747 - accuracy: 0.5000
## (45, 26, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
## 6/10 [=================>............] - ETA: 0s - loss: 1.5890 - accuracy: 0.3333
## (45, 43, 2) (45, 10)
## 7/10 [====================>.........] - ETA: 0s - loss: 1.4136 - accuracy: 0.4286
## (45, 8, 2) (45, 10)
##
## (45, 8, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 1.5031 - accuracy: 0.3333
## (45, 75, 2) (45, 10)
##10/10 [==============================] - 0s 38ms/step - loss: 1.5414 - accuracy: 0.3000
##Epoch 10/10
##
## (45, 37, 2) (45, 10)
## 1/10 [==>...........................] - ETA: 0s - loss: 0.1867 - accuracy: 1.0000
## (45, 75, 2) (45, 10)
##
## (45, 8, 2) (45, 10)
## 3/10 [========>.....................] - ETA: 0s - loss: 0.7578 - accuracy: 0.6667
## (45, 37, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
##
## (45, 18, 2) (45, 10)
## 6/10 [=================>............] - ETA: 0s - loss: 1.1529 - accuracy: 0.6667
## (45, 18, 2) (45, 10)
##
## (45, 75, 2) (45, 10)
##
## (45, 23, 2) (45, 10)
## 9/10 [==========================>...] - ETA: 0s - loss: 1.2356 - accuracy: 0.5556
## (45, 21, 2) (45, 10)
##10/10 [==============================] - 0s 35ms/step - loss: 1.2892 - accuracy: 0.5000
##Classify.py:130: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
##  X = [testX[i] for i in mask]
##Classify.py:131: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
##  y = [testY[i] for i in mask]
##(505, 37, 2)
##(505, 10)
##Sequence length: 1
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 2
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 3
##        Test loss: 1.1997145414352417
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 4
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 5
##        Test loss: 1.199713945388794
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 6
##        Test loss: 1.199713945388794
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 7
##        Test loss: 1.199713945388794
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 8
##        Test loss: 1.1997135877609253
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 9
##        Test loss: 1.1997138261795044
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 10
##        Test loss: 1.1997138261795044
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 11
##        Test loss: 1.1997135877609253
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 12
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 13
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 14
##        Test loss: 1.1997135877609253
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 15
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 16
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 17
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 18
##        Test loss: 1.1997135877609253
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 19
##        Test loss: 1.1997135877609253
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 20
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 21
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 22
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 23
##        Test loss: 1.1997138261795044
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 25
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 26
##        Test loss: 1.1997145414352417
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 27
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 28
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 30
##        Test loss: 1.199713945388794
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 31
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 32
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 33
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 35
##        Test loss: 1.1997145414352417
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 36
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 37
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 38
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 40
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 41
##        Test loss: 1.199714183807373
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 43
##        Test loss: 1.1997138261795044
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 44
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 46
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 48
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 52
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 53
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 55
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 56
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 58
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 62
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 64
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 65
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 68
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 69
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 74
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 76
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 88
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 91
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 93
##        Test loss: 1.1997145414352417
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 105
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 107
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 126
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 132
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 145
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0
##(505, 37, 2)
##(505, 10)
##Sequence length: 152
##        Test loss: 1.1997144222259521
##        Test accuracy: 1.0




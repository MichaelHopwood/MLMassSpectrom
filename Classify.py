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





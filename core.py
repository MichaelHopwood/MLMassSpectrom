import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from matplotlib.pyplot import cm
from sklearn.model_selection import train_test_split


SAVEPATH = 'figures'

os.environ['TF_KERAS'] = "1"  # configurable
os.environ['TF_EAGER'] = "0"  # configurable

print("TF version:", tf.__version__)

TF_2 = (tf.__version__[0] == '2')
eager_default = "1" if TF_2 else "0"
TF_EAGER = bool(os.environ.get('TF_EAGER', eager_default) == "1")
TF_KERAS = bool(os.environ.get('TF_KERAS', "0") == "1")

if TF_EAGER:
    if not TF_2:
        tf.enable_eager_execution()
    print("TF running eagerly")
else:
    if TF_2:
        tf.compat.v1.disable_eager_execution()
    print("TF running in graph mode")

tf.compat.v1.experimental.output_all_intermediates(True)

from see_rnn import get_gradients, get_outputs, get_rnn_weights
from see_rnn import features_0D, features_1D, features_2D
from see_rnn import rnn_heatmap, rnn_histogram

###############################################################################

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

def visualize_sample_distribution(X, y, filename='sample_distribution.png'):
    # Visualize the distribution of samples
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.colorbar()
    plt.savefig(os.path.join(SAVEPATH, filename))
    plt.show()


class Generator:
    """ Simulate spectrogram-like samples and vend these samples so that
    they are dispensed in groups of same sequence length.
    """
    def __init__(self, num_groups=2, num_samples_per_group=50, train_pct=0.9):
        self.num_groups = num_groups
        self.num_samples_per_group = num_samples_per_group
        self.train_pct = train_pct

        self.data_setup = {}
        self.numNonzero_to_groupIDs = {}
        # Save means of mass and intensity
        for group_id in range(num_groups):
            num_nonzero = generate_num_nonzero(1)[0]
            print('num_nonzero', num_nonzero)
            intensity_locs, mass_locs = get_locs(num_nonzero)
            self.data_setup[group_id] = {'num_nonzero': num_nonzero,
                                         'intensity_locs': intensity_locs,
                                         'mass_locs': mass_locs}
            if num_nonzero not in self.numNonzero_to_groupIDs.keys():
                self.numNonzero_to_groupIDs[num_nonzero] = []
            self.numNonzero_to_groupIDs[num_nonzero].append(group_id)

    def _generate_data(self, **generate_sample_kwargs):

        num_nonzero = np.random.choice(list(self.numNonzero_to_groupIDs.keys()))
        group_ids_iter = self.numNonzero_to_groupIDs[num_nonzero]
        X = []
        y = []
        for group_id in group_ids_iter:
            settings = self.data_setup[group_id]
            print('gen samples num_nonzero', num_nonzero)
            samples = generate_mass_samples(num_nonzero, self.num_samples_per_group,
                                            intensity_locs=settings['intensity_locs'],
                                            mass_locs=settings['mass_locs'],
                                            **generate_sample_kwargs)
            X.extend(samples)
            y.extend([group_id]*self.num_samples_per_group)
        return X, y

    def generate_data(self, **generate_sample_kwargs):
        X, y = self._generate_data(**generate_sample_kwargs)
        X = np.asarray(X)
        y = to_categorical(y, num_classes=self.num_groups)
        # Shuffle
        Xy = list(zip(X, y))
        random.shuffle(Xy)
        X, y = zip(*Xy)
        X=np.asarray(X)
        y=np.asarray(y)
        return X, y
    
    def generate_train_test_data(self, **generate_sample_kwargs):
        X, y = self.generate_data(**generate_sample_kwargs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.train_pct)

        import pandas as pd
        print('train iter y split:\n', pd.Series(np.argmax(y_train, axis=1)).value_counts())
        print('test iter y split:\n', pd.Series(np.argmax(y_test, axis=1)).value_counts())

        y_train = to_categorical(y_train, num_classes=self.num_groups)
        y_test = to_categorical(y_test, num_classes=self.num_groups)
        return X_train, X_test, y_train, y_test

    def train_generator(self):
        self.testX, self.testY, self.testLength = [], [], []
        while True:
            sequence_length = generate_num_nonzero(1, lognorm_mean=3, lognorm_sigm=1)[0]

            # Split into train-test splits
            trainX, testX, trainy, testy = self.generate_train_test_data()

            self.testX.extend(testX)
            self.testY.extend(testy)
            self.testLength.extend([sequence_length]*len(testy))
            yield trainX, trainy


class VisualizeRNNLayers(Generator):
    def __init__(self):
        super().__init__()
    
    def _filename(self, name):
        return os.path.join(self.savepath, name)

    def visualize(self, model, savepath):
        self.savepath = SAVEPATH
        self.viz_outs(model, idx='encoder')
        self.viz_weights(model, idx='encoder')
        self.viz_outs_grads(model, idx='encoder')
        self.viz_outs_grads_last(model, idx='reducer')
        self.viz_weights_grads(model, idx='encoder')

        data = get_rnn_weights(model, "encoder")
        self.viz_prefetched_data(model, data, idx='encoder')

    def viz_outs(self, model, idx=1):
        x, y = self.generate_data()
        outs = get_outputs(model, idx, x)

        features_1D(outs[:1], n_rows=8, show_borders=False, savepath=self._filename('outs_1D_'+str(idx)))
        features_2D(outs,     n_rows=8, norm=(-1,1), savepath=self._filename('outs_2D_'+str(idx)))

    def viz_weights(self, model, idx=1):
        rnn_histogram(model, idx, mode='weights', bins=400, savepath=self._filename('weights_'+str(idx)))
        print('\n')
        rnn_heatmap(model,   idx, mode='weights', norm='auto', savepath=self._filename('weights_heatmap_'+str(idx)))

    def viz_outs_grads(self, model, idx=1):
        x, y = self.generate_data()
        grads = get_gradients(model, idx, x, y)
        kws = dict(n_rows=8, title='grads')

        features_1D(grads[0], show_borders=False, **kws, savepath=self._filename('outs_grads_1D_'+str(idx)))
        features_2D(grads,    norm=(-1e-4, 1e-4), **kws, savepath=self._filename('outs_grads_2D_'+str(idx)))

    def viz_outs_grads_last(self, model, idx=2):  # return_sequences=False layer
        x, y = self.generate_data()
        grads = get_gradients(model, idx, x, y)
        features_0D(grads, savepath=self._filename('outs_grads_last_'+str(idx)))

    def viz_weights_grads(self, model, idx=1):
        x, y = self.generate_data()
        kws = dict(_id=idx, input_data=x, labels=y, savepath=self._filename('weights_grads_'+str(idx)))

        rnn_histogram(model, mode='grads', bins=400, **kws)
        print('\n')

        kws.update({'savepath': self._filename('weights_grads_heatmap_'+str(idx))})
        rnn_heatmap(model,   mode='grads', cmap=None, absolute_value=True, **kws)

    def viz_prefetched_data(self, model, data, idx=1):
        rnn_histogram(model, idx, data=data, savepath=self._filename('prefetched_data_'+str(idx)))
        rnn_heatmap(model,   idx, data=data, savepath=self._filename('prefetched_data_heatmap_'+str(idx)))

class NN:
    def __init__(self):
        self.gen = Generator()

    def make_model(self):
        self.model = Sequential()
        self.model.add(LSTM(4, return_sequences=True, input_shape=(None, 2), name="encoder"))
        self.model.add(LSTM(12, return_sequences=False, name='reducer'))
        #self.model.add(Flatten())
        #self.model.add(TimeDistributed(Flatten()))
        #self.model.add(Flatten())
        self.model.add(Dense(self.gen.num_groups, activation='sigmoid'))
        #self.model.add(TimeDistributed(Dense(10, activation='sigmoid')))

        print(self.model.summary())
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=1e-2), metrics=['accuracy'])

    def train(self, epochs=10):
        self.model.fit(self.gen.train_generator(), steps_per_epoch=20, epochs=epochs, verbose=1)

        #batch_shape = K.int_shape(self.model.input)
        #units = self.model.layers[2].units
        # x, y = next(self.gen.train_generator())
        # print("DONE", x.shape, y.shape)

        # for i in range(iterations):
        #     self.model.train_on_batch(x, y)
        #     print(end='.')  # progbar
        #     if i % 10 == 0:
        #         x, y = next(self.gen.train_generator())


    def evaluate(self):
        testLength = np.array(self.gen.testLength)

        def _filename(name):
            return os.path.join(SAVEPATH, name)

        fig, ax = plt.subplots()
        # (B, ?, 2)
        X = self.gen.testX
        # (B, self.gen.num_groups)

        y = self.gen.testY
        print(np.array(X).shape)
        print(np.array(X[0]).shape)

        y = np.argmax(y, axis=1)

        import pandas as pd
        print(pd.Series(y).value_counts())

        color = iter(cm.rainbow(np.linspace(0, 1, self.gen.num_groups)))
        for i in range(self.gen.num_groups):
            c = next(color)
            mask = y == i

            # (?, ?, 2)
            X = [self.gen.testX[i] for i in mask]
            print(i)
            print(np.array(X).shape)
            print(np.array(X[0]).shape)
            print(c)
            for x in X:
                for xi in x:
                    plt.axvline(xi[0], ymax=xi[1], c=c)
        plt.savefig(_filename('test_data.png'))
        plt.show()

        sequence_length = []
        accuracies = []
        losses = []
        for sequence_len in np.unique(testLength):
            mask = testLength == sequence_len
            X = [self.gen.testX[i] for i in mask]
            y = [self.gen.testY[i] for i in mask]
            X = np.array(X)
            y = np.array(y)
            score = self.model.evaluate(X, y, verbose=0)
            # print('Sequence length:', sequence_len)
            # print('\tTest loss:', score[0])
            # print('\tTest accuracy:', score[1])

            sequence_length.append(sequence_len)
            accuracies.append(score[1])
            losses.append(score[0])


        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(sequence_length, accuracies, 'g-')
        ax2.plot(sequence_length, losses, 'b-')

        ax1.set_xlabel('Sequence length')
        ax1.set_ylabel('Accuracy', color='g')
        ax2.set_ylabel('Loss', color='b')
        plt.savefig(_filename('accuracy_loss.png'))
        plt.show()
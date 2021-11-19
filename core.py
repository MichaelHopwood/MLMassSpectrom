from matplotlib import patches
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, GlobalAveragePooling1D, AveragePooling2D, MultiHeadAttention, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Reshape, Input, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
import os
from matplotlib.pyplot import cm
from attention import Attention

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

def generate_num_nonzero(lognorm_mean=3, lognorm_sigm=1):
    # Select number of nonzero terms in spectroscopy sample
    # Ensure it is nonzero
    sample = int(np.random.lognormal(lognorm_mean, lognorm_sigm))
    if sample == 0:
        return generate_num_nonzero(lognorm_mean, lognorm_sigm)
    else:
        return sample

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

def visualize_sample_distribution(X, y, filepathname='.//sample_distribution.png'):
    # Visualize the distribution of samples
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.colorbar()
    plt.savefig(filepathname)
    plt.close()


class Generator:
    """ Simulate spectrogram-like samples and vend these samples so that
    they are dispensed in groups of same sequence length.
    """
    def __init__(self, num_groups=2, num_samples_per_group=50, train_pct=0.7, valid_pct=0.1, case=None, savepath=None):
        self.num_groups = num_groups
        self.num_samples_per_group = num_samples_per_group
        self.train_pct = train_pct
        self.valid_pct = valid_pct
        self.case = case
        self.savepath = savepath

        if isinstance(self.case, type(None)):
            self._init_random_experiment()
        else:
            self._case_experiment(self.case)

    def _case_experiment(self, case_id):

        if case_id == 0:
            self.num_groups = 2
            self.data_setup = {}
            self.numNonzero_to_groupIDs = {3: [0], 2: [1]}
            # Save means of mass and intensity
            self.data_setup[0] = {'num_nonzero': 3, 'intensity_locs': [0.25, 0.5, 0.25], 'mass_locs': [50, 150, 250]}
            self.data_setup[1] = {'num_nonzero': 2, 'intensity_locs': [0.5, 0.5], 'mass_locs': [100, 200]}

        elif case_id == 1:
            self.num_groups = 2
            self.data_setup = {}
            self.numNonzero_to_groupIDs = {2: [0, 1]}
            # Save means of mass and intensity
            self.data_setup[0] = {'num_nonzero': 2, 'intensity_locs': [0.5, 0.5], 'mass_locs': [50, 100]}
            self.data_setup[1] = {'num_nonzero': 2, 'intensity_locs': [0.5, 0.5], 'mass_locs': [150, 200]}

        elif case_id == 2:
            self.num_groups = 2
            self.data_setup = {}
            self.numNonzero_to_groupIDs = {2: [0, 1]}
            # Save means of mass and intensity
            self.data_setup[0] = {'num_nonzero': 2, 'intensity_locs': [0.1, 0.3], 'mass_locs': [50, 100]}
            self.data_setup[1] = {'num_nonzero': 2, 'intensity_locs': [0.7, 0.9], 'mass_locs': [150, 200]}

        elif case_id == 3:
            self.num_groups = 2
            self.data_setup = {}
            self.numNonzero_to_groupIDs = {2: [0], 14: [1]}
            # Save means of mass and intensity
            self.data_setup[0] = {'num_nonzero': 2, 'intensity_locs': [0.1, 0.3], 'mass_locs': [50, 100]}
            ilocs = [0.05, 0.09, 0.3, 0.05, 0.81, 0.74, 0.1, 0.04, 0.7, 0.9, 0.05, 0.1, 0.5, 0.13]
            mlocs = [14, 134, 143, 132, 135, 142, 133, 211, 214, 284, 245, 246, 255, 289]
            self.data_setup[1] = {'num_nonzero': 2, 'intensity_locs': ilocs, 'mass_locs':mlocs}

        else:
            raise ValueError("Case ID not recognized")

    def _init_random_experiment(self):
        self.data_setup = {}
        self.numNonzero_to_groupIDs = {}
        # Save means of mass and intensity
        for group_id in range(self.num_groups):
            num_nonzero = generate_num_nonzero()
            intensity_locs, mass_locs = get_locs(num_nonzero)
            self.data_setup[group_id] = {'num_nonzero': num_nonzero,
                                         'intensity_locs': intensity_locs,
                                         'mass_locs': mass_locs}
            if num_nonzero not in self.numNonzero_to_groupIDs.keys():
                self.numNonzero_to_groupIDs[num_nonzero] = []
            self.numNonzero_to_groupIDs[num_nonzero].append(group_id)

    def generate_data(self, **generate_sample_kwargs):
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

    def train_generator(self):
        self.testX, self.testY = [], []
        self.validX, self.validY = [], []
        self.testLength = []
        self.validLength = []
        while True:
            X, y = self.generate_data()
            # Split into train-test splits
            train_split = int(self.train_pct * len(y))
            valid_split = int((self.train_pct + self.valid_pct) * len(y))

            trainX, trainy = X[:train_split], y[:train_split]
            self.testX.extend(X[train_split:valid_split])
            self.testY.extend(y[train_split:valid_split])
            self.validX.append(X[valid_split:])
            self.validY.append(y[valid_split:])
            self.testLength.extend([X.shape[1]] * len(y[train_split:valid_split]))
            self.validLength.extend([X.shape[1]] * len(y[valid_split:]))
            yield trainX, trainy

    def validation_generator(self):
        while True:
            random_index = random.randrange(len(self.validX))
            yield self.validX[random_index], self.validY[random_index]

class VisualizeRNNLayers(Generator):
    def __init__(self):
        super().__init__()
    
    def _filename(self, name):
        return os.path.join(self.savepath, name)

    def visualize(self, model, savepath):
        self.savepath = savepath
        self.viz_outs(model, idx='encoder')
        self.viz_weights(model, idx='encoder')
        self.viz_outs_grads(model, idx='encoder')
        # self.viz_outs_grads_last(model, idx='reducer')
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

from keract import get_activations
from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model, Model
from tensorflow.python.keras.utils.vis_utils import plot_model

class NN:
    def __init__(self, nn_type='lstm', **gen_kwargs):
        self.gen = Generator(**gen_kwargs)
        self.nn_type = nn_type

    def _filename(self, name):
        return os.path.join(self.gen.savepath, name)

    def make_model(self, lstm_hidden_size=32):

        if self.nn_type == 'lstm':
            self.model = Sequential()
            self.model.add(LSTM(lstm_hidden_size, return_sequences=True, input_shape=(None, 2), name="encoder"))
            
            self.model.add(GlobalAveragePooling1D(name='reducer'))
            # self.model.add(Attention(name='reducer'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.gen.num_groups, activation='softmax'))


            print(self.model.summary())
            self.model.compile(loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate=1e-2), metrics=['accuracy'])

        elif self.nn_type == 'attention':
            self.model = Sequential()
            #self.model.add(MultiHeadAttention(input_shape=(None, 2), num_heads=2, key_dim=2, value_dim=2, name='encoder'))
            self.model.add(Attention(input_shape=(None, 15, 2), name='encoder'))
            self.model.add(Dense(self.gen.num_groups, activation='softmax'))

        plot_model(self.model, to_file=self._filename('model.png'), show_shapes=True)

    def train(self, epochs=25):
        x_one_sample, y_one_sample = next(self.gen.train_generator())

        savepath = self.gen.savepath
        class VisualiseAttentionMap(Callback):
            def on_epoch_end(self, epoch, logs=None):
                try:
                    attention_map = get_activations(self.model, x_one_sample)['attention_weight']
                    # top is attention map, bottom is ground truth.
                    plt.imshow(attention_map, cmap='hot')
                    iteration_no = str(epoch).zfill(3)
                    plt.axis('off')
                    plt.title(f'Iteration {iteration_no} / {epochs}')
                    plt.savefig(f'{savepath}/epoch_{iteration_no}.png')
                    plt.close()
                except KeyError:
                    # For models with no attention layer
                    pass

        self.history = self.model.fit(self.gen.train_generator(),
                                      steps_per_epoch=50,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=self.gen.validation_generator(),
                                      validation_steps=100,
                                      callbacks=[VisualiseAttentionMap()])

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
        validLength = np.array(self.gen.validLength)

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        ax1.plot(list(range(len(testLength))), testLength)
        ax2.plot(list(range(len(validLength))), validLength)
        ax1.set_xlabel('Iteration')
        ax2.set_xlabel('Iteration')
        ax1.set_ylabel('Test Length')
        ax2.set_ylabel('Validation Length')
        plt.savefig(self._filename('testLength.png'))
        plt.close()
        # self.gen.testX = np.array(self.gen.testX)
        # self.gen.testY = np.array(self.gen.testY)

        fig, ax = plt.subplots()
        # (B, ?, 2)
        X = self.gen.testX
        # (B, self.gen.num_groups)
        y = self.gen.testY
        # (B,)
        y = np.argmax(y, axis=1)


        if False:
            # Takes a long time to execute
            color = iter(cm.rainbow(np.linspace(0, 1, self.gen.num_groups)))
            for i in range(self.gen.num_groups):
                c = next(color)
                mask = np.where(y == i)[0]
                Xi = np.array([X[i] for i in mask])
                print(Xi.shape)
                reshape_Xi = Xi.reshape(Xi.shape[0]*Xi.shape[1], Xi.shape[2])
                plt.bar(reshape_Xi[:,0], reshape_Xi[:,1], color=c)

            plt.savefig(self._filename('test_data.png'))
            plt.close()

        sequence_length = []
        accuracies = []
        losses = []
        for sequence_len in np.unique(testLength):
            mask = np.where(testLength == sequence_len)[0]
            X = [self.gen.testX[i] for i in mask]
            y = [self.gen.testY[i] for i in mask]
            X = np.array(X)
            y = np.array(y)
            score = self.model.evaluate(X, y, verbose=0)

            sequence_length.append(sequence_len)
            accuracies.append(score[1])
            losses.append(score[0])


        fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.scatter(sequence_length, accuracies, color='r')
        ax2.scatter(sequence_length, losses, color='r')

        ax1.set_xlabel('Sequence length')
        ax2.set_xlabel('Sequence length')
        ax1.set_ylabel('Accuracy')
        ax2.set_ylabel('Loss')
        fig.tight_layout()
        plt.savefig(self._filename('test_accuracy_loss.png'))
        plt.close()

        print(self.history.history)

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='best')
        plt.savefig(self._filename('train_accuracy.png'))
        plt.close()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='best')
        plt.savefig(self._filename('train_loss.png'))
        plt.close()  

        return accuracies, losses, testLength
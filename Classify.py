import numpy as np
from core import NN, VisualizeRNNLayers
from pathlib import Path
import os
import pandas as pd

# DEFAULTS
# These are often overwritten when experimenting
nn_type = 'lstm'
NUM_RANDOM_GROUPS = 10 # Only used if CASE=None
NUM_SAMPLES_PER_GROUP = 50
CASE = None # None, 0, 1, 2, 3. None=random

LSTM_HIDDEN_SIZE = 32

TRAIN_PCT = 0.7
VALID_PCT = 0.1

class Experimenter:
    def __init__(self):
        pass

    def _build_generator_kwargs(self,
                                num_groups=NUM_RANDOM_GROUPS,
                                num_samples_per_group=NUM_SAMPLES_PER_GROUP,
                                train_pct=TRAIN_PCT,
                                valid_pct=VALID_PCT,
                                case=CASE):

        savepath = os.path.join('figures', self.experiment_save_identifier, f'iter{self._iter}')

        Path(savepath).mkdir(parents=True, exist_ok=True)

        # Establish generator defaults
        self.gen_kwargs = {'num_groups':num_groups,
                    'num_samples_per_group':num_samples_per_group,
                    'train_pct':train_pct,
                    'valid_pct':valid_pct,
                    'case':case,
                    'savepath':savepath}
    
    def _build_model_kwargs(self,
                            lstm_hidden_size=LSTM_HIDDEN_SIZE):
        # Establish NN model default
        self.model_kwargs = {'lstm_hidden_size':lstm_hidden_size}

    def _run(self):
        self.nn = NN(nn_type, **self.gen_kwargs)
        self.nn.make_model(**self.model_kwargs)
        self.nn.train()
        results = self.nn.evaluate()
        # Accuracy per unique sequence length
        self.accuracies = results[0]
        # Losses per unique sequence length
        self.losses = results[1]

        #vz = VisualizeRNNLayers()
        #vz.visualize(nn.model, self.gen_kwargs['savepath'])

    def _save(self):
        self.experiment_results.loc[self._iter] = [self.gen_kwargs['num_groups'],
                                                   self.model_kwargs['lstm_hidden_size'],
                                                   self.accuracies,
                                                   self.losses,
                                                   self.nn.gen.data_setup]
        self._iter += 1
        self.experiment_results.to_csv(os.path.join(self.experiment_savepath, 'experiment_results.csv'), index=False)

    def _experiment_numGroups_vs_LSTMHiddenSize(self):
        for num_groups in np.arange(2, 100, 2):
            for lstm_size in [16, 32, 64, 128, 256]:
                num_groups = int(num_groups)

                # Update experiment settings
                self._build_generator_kwargs(num_groups=num_groups)
                self._build_model_kwargs(lstm_hidden_size=lstm_size)

                # Run experiment iteration and save results
                self._run()
                self._save()

    def experiment(self, experiment_type='numGroups_vs_LSTMHiddenSize', experiment_save_identifier='numGroups_vs_LSTMHiddenSize'):
        
        # Prep for experiment
        self._iter = 0
        self.experiment_results = pd.DataFrame(columns=['num_groups', 'lstm_hidden_size', 'accuracies', 'losses', 'data_setup'])
        self.experiment_save_identifier = experiment_save_identifier

        # Establish experiment folder
        self.experiment_savepath = os.path.join('figures', self.experiment_save_identifier)
        Path(self.experiment_savepath).mkdir(parents=True, exist_ok=True)

        # Run experiment
        if experiment_type == 'numGroups_vs_LSTMHiddenSize':
            self._experiment_numGroups_vs_LSTMHiddenSize()
        else:
            raise Exception(f'Experiment type {experiment_type} not recognized.')


if __name__ == '__main__':
    exp = Experimenter()
    exp.experiment(experiment_save_identifier='numGroups_vs_LSTMHiddenSize_withReducerAveragePoolingAndDropout')

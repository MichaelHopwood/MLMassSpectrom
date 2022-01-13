import numpy as np
from core import NN, VisualizeRNNLayers, RealDataGenerator, SimulationGenerator, BothSimAndRealGenerator
from pathlib import Path
import os
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

###########################
## Real data processing ##
###########################
def build_data_structure(df, labels, column_name_mass_locations, append=True, gt=0):
    """
    Convert dataframe which has columns of the mass locations and values of the intensities
    to array format where the sample is a list of peaks and the samples are grouped by the number of peaks.

    Input:
    df: dataframe with columns of the mass locations and values of the intensities
    	Col0  Col1  Col2 ...
    0	0.0	  0.0	0.0

    column_name_mass_locations: column names of the mass locations
    [59.00498,
     72.00792,
     74.00967,
     ...
    ]

    We split it up just in case duplicate mass locations are present since multiple scans can be represented.

    Output:
    [ [samples with smallest number of peaks],
      [samples with 2nd smallest number peaks], 
      ...,
      [samples with largest number of peaks]
    ]

    where each sample is an array of lists (mass, intensity):
    array([[peak_location1, peak_intensity1], [peak_location2, peak_intensity2], ...])

    Specify `append=True` to create separate lists for each peak size. Otherwise, all samples are extended along same axis.
    """
    df = df.copy()
    # Get nonzero values (aka "peaks")
    data = df.apply(lambda x: np.asarray([[column_name_mass_locations[i], val] for i, (val, b) in enumerate(zip(x, x.gt(gt))) if b]), axis=1)    

    X = []
    Y = []
    # Group so we have groups of the same number of peaks
    lengths = np.array([len(x) for x in data])
    unique_lengths = np.unique(lengths)
    for length in unique_lengths:
        mask_length = lengths == length
        mask_idx = np.where(mask_length)[0]
        y = labels[mask_idx]
        x = np.stack(data.loc[mask_length].values.tolist())
        if append:
            X.append(x)
            Y.append(y)
        else:
            X.extend(x)
            Y.extend(y)
    return X, Y

def prepare_FullDartData(df):
    labeled_df = df[df['Sample Types'] == 'Model']

    column_name_mass_locations = [(float('.'.join(col.split('.')[:2])) if isinstance(col, str) else col) for col in df.columns[4:]]

    lb = preprocessing.LabelBinarizer()
    lb.fit(labeled_df['Class'].unique())

    train_labeled_df, test_labeled_df = model_selection.train_test_split(labeled_df, train_size=TRAIN_PCT, shuffle=True, stratify=labeled_df['Class'])

    X_train, y_train = build_data_structure(train_labeled_df[train_labeled_df.columns[4:]],
                                   train_labeled_df['Class'].values,
                                   column_name_mass_locations, gt=gt)
    X_valid, y_valid = build_data_structure(test_labeled_df[test_labeled_df.columns[4:]],
                                  test_labeled_df['Class'].values,
                                  column_name_mass_locations, gt=gt)
    X_test, y_test = build_data_structure(test_labeled_df[test_labeled_df.columns[4:]],
                                  test_labeled_df['Class'].values,
                                  column_name_mass_locations,
                                  append=False, gt=gt)
    y_train = [lb.transform(y) for y in y_train]
    y_valid = [lb.transform(y) for y in y_valid]
    y_test = lb.transform(y_test)

    num_classes = len(lb.classes_)
    return X_train, X_test, X_valid, y_train, y_test, y_valid, num_classes

if __name__ == '__main__':
    nn_type = 'lstm'
    TRAIN_PCT = 0.7
    LSTM_HIDDEN_SIZE = 512
    NUM_LSTM_LAYERS = 3
    save_folder = 'figures_real_data'
    experiment_save_identifier = 'test'
    _iter = 1
    gt = 25 # threshold for all values above this weight are considered as a peak

    folder = 'data'
    df = pd.read_excel(os.path.join(folder, '20211128 - Full DART Data (Model & Test).xlsx'), header=2)
    print(df.head())

    X_train, X_test, X_valid, y_train, y_test, y_valid, num_classes = prepare_FullDartData(df)

    print(f'X_train[0].shape: {X_train[0].shape}')
    print(f'X_test[0].shape: {X_test[0].shape}')
    print(f'X_valid[0].shape: {X_valid[0].shape}')
    print(f'y_train[0].shape: {y_train[0].shape}')
    print(f'y_test[0].shape: {y_test[0].shape}')
    print(f'y_valid[0].shape: {y_valid[0].shape}')

    savepath = os.path.join(save_folder, experiment_save_identifier, f'iter{_iter}')
    Path(savepath).mkdir(parents=True, exist_ok=True)

    realdata_gen_kwargs = dict(zip(
        ['trainX', 'trainy', 'testX', 'testY', 'validX', 'validY', 'num_groups', 'savepath'], [X_train, y_train, X_test, y_test, X_valid, y_valid, num_classes, savepath]
    ))
    realDataGenerator = RealDataGenerator(**realdata_gen_kwargs)

    model_kwargs = {'lstm_hidden_size':LSTM_HIDDEN_SIZE,
                    'num_lstm_layers':NUM_LSTM_LAYERS}
    
    simulationGenerator = SimulationGenerator(case='manual')
    sim_gen_kwargs = {'case': 'manual'}
    simulationGenerator.setVar(sim_gen_kwargs)

            self.num_groups = 2
            self.data_setup = {}
            self.numNonzero_to_groupIDs = {3: [0], 2: [1]}
            # Save means of mass and intensity
            self.data_setup[0] = {'num_nonzero': 3, 'intensity_locs': [0.25, 0.5, 0.25], 'mass_locs': [50, 150, 250]}

    gen_kwargs = {'simulationGenerator': simulationGenerator,
                  'realDataGenerator': realDataGenerator}

    nn = NN(nn_type, generator=BothSimAndRealGenerator, **gen_kwargs)
    nn.make_model(**model_kwargs)
    nn.train()
    results = nn.evaluate()
    print(results)

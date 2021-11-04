import numpy as np
import matplotlib.pyplot as plt


"""Generate number of nonzero features"""

def generate_num_nonzero(num_samples, lognorm_mean=3, lognorm_sigm=1):
    # Select number of nonzero terms in spectroscopy sample
    return np.random.lognormal(lognorm_mean, lognorm_sigm, num_samples).astype(int)


"""Generate Mass Samples"""

def generate_mass_samples(num_nonzero, num_samples,
                          low=0, high=300, mass_std=0.001,
                          intensity_spread_factor=1.0, intensity_std=0.01):
    # Generate the location of the num_nonzero components
    #intensity_locs = np.random.dirichlet(np.ones(num_nonzero)*float(intensity_spread_factor))
    counts = np.random.uniform(0, 1, num_nonzero)
    intensity_locs = counts / counts.sum()
    
    #print("intensity means:", intensity_locs)
    samples = []
    for i in range(num_nonzero):
        loc = np.random.uniform(low, high)
        mass_samples = np.random.normal(loc, mass_std, num_samples)
        intensity_samples = np.random.normal(intensity_locs[i], intensity_std, num_samples)
        samples.append(np.column_stack((mass_samples, intensity_samples)))
    samples = np.array(samples)
    #return np.hstack(samples)
    samples[samples < 0] = 0
    return samples

def generate_sample_group(num_samples, lognorm_mean=3, lognorm_sigm=1,
                         low=0, high=300, mass_std=0.001,
                         intensity_spread_factor=1.0, intensity_std=0.01):
    num_nonzero = generate_num_nonzero(1, lognorm_mean=3, lognorm_sigm=1)[0]
    samples = generate_mass_samples(num_nonzero, num_samples,
                              low=low, high=high, mass_std=mass_std,
                              intensity_spread_factor=intensity_spread_factor, intensity_std=intensity_std)
    return samples

def visualize_samples(samples):
    for sample in samples:
        mass, intensity = sample[:,0], sample[:,1]
        plt.bar(mass, intensity)
    plt.show()

    
def generate_all_samples(ngroups=100, nsamples_in_group=10, **kwargs):
    """Generates samples for multiple test groups.
    
    Returns
    -------
    X : `ngroup` size list of arrays: (nsamples_in_group, random_num_nonzero, 2)
    y : `ngroup`*`nsamples_in_group` size list of strings
    """
    X = []
    y = []
    for group_i in range(ngroups):
        samples = generate_sample_group(nsamples_in_group, **kwargs)
        n_samples, _, _ = samples.shape
        reshape_samples = np.reshape(samples, (nsamples_in_group, n_samples, 2))
        #visualize_samples(samples)
        #print(reshape_samples.shape)
        X.extend(reshape_samples)
        y.extend([group_i]*nsamples_in_group)
    return X, y


import numpy as np
import scipy.io as scio
from IPython import embed
import matplotlib.pyplot as plt


def read_data(file_name):
    data = scio.loadmat(file_name)
    spike_times = []
    spikes = data["responses_strong"]
    t = np.arange(0, 10., 1./20000)
    for i in range(spikes.shape[1]):
        spike_times.extend(list(t[spikes[:, i] == 1]))
    spike_times = np.asarray(spike_times)
    return spike_times, len(spike_times)


def estimate_optimal_binwidth(spike_times, trial_count):
    dts =  np.arange(0.00025, 0.05, 0.00001)
    c_ns = np.zeros(dts.shape)
    for i, dt in enumerate(dts):
        bin_edges = np.arange(dt, 10., dt)
        n, e = np.histogram(spike_times, bin_edges)
        k = np.mean(n)
        v = np.mean((n-k)**2)
        c_n = (2 * k - v)/((trial_count * dt)**2)
        c_ns[i] = c_n

    plt.plot(dts*1000, c_ns)
    plt.xlabel("bin_width [ms]")
    plt.ylabel(r"$C_n$")
    plt.show()
    return dts[np.nonzero(c_ns == np.min(c_ns))[0]]


def do_the_kde():
    spike_times, trial_count = read_data('../data/data_p-unit.mat')
    estimate_optimal_binwidth(spike_times, trial_count)
    embed()
    exit()


if __name__ == "__main__":
    do_the_kde()

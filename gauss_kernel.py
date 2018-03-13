import numpy as np

def gauss_kernel(sigma, sample_rate, duration):
    """
    Creates a Gaussian kernel centered in a vector of given duration.

    :param sigma: the standard deviation of the kernel in seconds
    :param sample_rate: the temporal resolution of the kernel in Hz
    :param duration: the desired duration of the kernel in seconds, (in general at least 4 * sigma)

    :return: the kernel as numpy array
    """
    l = duration * sample_rate
    x = np.arange(-np.floor(l / 2), np.floor(l / 2)) / sample_rate
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x ** 2 / (2 * sigma ** 2)))
    y /= np.sum(y)
    return y

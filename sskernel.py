import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from IPython import embed

def ilogexp(x):
    if x < 1e2:
        y = np.log(np.exp(x)-1)
    else:
        y = x
    return y


def logexp(x):
    if x < 1e2:
        y = np.log(1 + np.exp(x))
    else:
        y = x
    return y


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def fftkernel(x,w):
    """
    % y = fftkernel(x,w)
    %
    % Function `fftkernel' applies the Gauss kernel smoother to 
    % an input signal using FFT algorithm.
    %
    % Input argument
    % x:    Sample signal vector. 
    % w: 	Kernel bandwidth (the standard deviation) in unit of 
    %       the sampling resolution of x. 
    %
    % Output argument
    % y: 	Smoothed signal.
    %
    % MAY 5/23, 2012 Author Hideaki Shimazaki
    % RIKEN Brain Science Insitute
    % http://2000.jukuin.keio.ac.jp/shimazaki
    """
    L = len(x)
    Lmax = np.max(np.arange(1, L+3*w))
    n = int(nextpow2(Lmax))

    X = np.fft.fft(x,n)
    f = np.arange(n)/float(n)
    f = np.hstack((-f[:n/2+1], f[n/2-1:0:-1]))

    K = np.exp(-0.5*(w*2*np.pi*f)**2)
    y = np.abs(np.fft.ifft(X*K, n))
    y = y[:L]
    return y


def cost_function(y_hist, N, w, dt):
    yh = fftkernel(y_hist, w/dt)
    C = np.sum(yh**2) * dt - 2 * np.sum(yh * y_hist) * dt + 2 * 1/np.sqrt(2*np.pi)/w/N    
    C = C * N * N

    return C, yh


def sskernel(x, t_in=None, widths=None):
    """
    % [y,t,optw] = sskernel(x,t,W)
    %
    % Function `sskernel' returns an optimized kernel density estimate 
    % using a Gauss kernel function.
    %
    % Examples:
    % >> x = 0.5-0.5*log(rand(1,1e3)); t = linspace(0,3,1000);
    % >> [y,t,optw] = sskernel(x,t);
    % This example produces a vector of kernel density estimates, y, at points
    % specified in a vector t, using an optimized bandwidth, optw (a standard 
    % deviation of a normal density function).
    % 
    % >> sskernel(x);
    % By calling the function without output arguments, the estimated density 
    % is displayed along with 95% bootstrap confidence intervals.
    %
    % Input arguments:
    % x:    Sample data vector. 
    % t_in (optinal):
    %       Points at which estimation are computed. Please use fine resolution
    %       to obtain a correct optimal bandwidth.
    % W (optinal): 
    %       A vector of kernel bandwidths. 
    %       If W is provided, the optimal bandwidth is selected from the 
    %       elements of W.
    %       * Do not search bandwidths smaller than a sampling resolution of data.
    %       If W is not provided, the program searches the optimal bandwidth
    %       using a golden section search method. 
    %
    % Output arguments:
    % y:    Estimated density
    % t:    Points at which estimation was computed.
    %       The same as tin if tin is provided. 
    %       (If the sampling resolution of tin is smaller than the sampling 
    %       resolution of the data, x, the estimation was done at smaller
    %       number of points than t. The results, t and y, are obtained by 
    %       interpolating the low resolution sampling points.)
    % optw: Optimal kernel bandwidth.
    % 
    % Usage:
    % >> [y,t,optw] = sskernel(x);
    % When t is not given in the input arguments, i.e., the output argument t 
    % is generated automatically.
    %
    % Optimization principle:
    % The optimal bandwidth is obtained as a minimizer of the formula, 
    % sum_{i,j} \int k(x - x_i) k(x - x_j) dx  -  2 sum_{i~=j} k(x_i - x_j), 
    % where k(x) is the kernel function, according to
    %
    % Hideaki Shimazaki and Shigeru Shinomoto
    % Kernel Bandwidth Optimization in Spike Rate Estimation 
    % Journal of Computational Neuroscience 2010
    % http://dx.doi.org/10.1007/s10827-009-0180-4
    %
    % The above optimization is based on a principle of minimizing 
    % expected L2 loss function between the kernel estimate and an unknown 
    % underlying density function. An assumption is merely that samples 
    % are drawn from the density independently each other. 
    %
    % For more information, please visit 
    % http://2000.jukuin.keio.ac.jp/shimazaki/res/kernel.html
    %
    % See also SSVKERNEL, SSHIST
    % 
    % Bug fix
    % 131004 fixed a problem for large values
    %
    % Hideaki Shimazaki 
    % http://2000.jukuin.keio.ac.jp/shimazaki
    """
    x = np.reshape(x, (1, np.size(x)))
                   
    if t_in == None:
        T = np.max(x) - np.min(x)
        tmp = np.sort(np.diff(np.sort(x)))
        mbuf,nbuf = np.nonzero(tmp)
        dt_samp = tmp[mbuf[0], nbuf[0]]
        t_in = np.linspace(np.min(x), np.max(x), np.min([np.ceil(T/dt_samp), 1e3]))
        t = t_in
        x_ab = x[(x >= np.min(t_in)) & (x <= np.max(t_in))]
    else:
        T = np.max(t_in) - np.min(t_in);    
        x_ab = x[(x >= np.min(t_in)) & (x <= np.max(t_in))]
        tmp = np.sort(np.diff(np.sort(x)))
        mbuf,nbuf = np.nonzero( tmp)
        dt_samp = tmp[mbuf, nbuf]
        if dt_samp > np.min(np.diff(t_in)):
            t = np.linspace(np.min(t_in), np.max(t_in), np.min([np.ceil(T/dt_samp), 1e3]))
        else:
            t = t_in;
    dt = np.min(np.diff(t))

    # Create a finest histogram
    y_hist,_ = np.histogram(x_ab, t-dt/2)
    L = len(y_hist)
    N = np.sum(y_hist)
    y_hist = np.asarray(y_hist, dtype=np.float)/N/dt

    if widths is not None:
        #  Global search
        C = np.zeros((1, len(widths)))
        C_min = np.Inf
        for i, w in enumerate(widths):
            C[i], yh = cost_function(y_hist, N, w, dt)
            if C(i) < C_min:
                C_min = C(i)
                optw = w
                y = yh
    else:
        # Golden section search on a log-exp scale
        Wmin = 2 * dt
        Wmax = 1 * (np.max(x) - np.min(x))
        tol = 10^-5;
        phi = (np.sqrt(5) + 1)/2        # golden ratio
        a = ilogexp(Wmin)
        b = ilogexp(Wmax)

        c1 = (phi-1)*a + (2-phi)*b
        c2 = (2-phi)*a + (phi-1)*b

        f1,_ = cost_function(y_hist, N, logexp(c1), dt)
        f2,_ = cost_function(y_hist, N, logexp(c2), dt)

        k = 1
        W = []
        C = []
        while np.abs(b-a) > tol * (np.abs(c1) + np.abs(c2)) and k <= 20:
            if (f1 < f2):    
                b = c2
                c2 = c1
                c1 = (phi - 1)*a + (2 - phi)*b
                f2 = f1
                f1, yh1 = cost_function(y_hist, N, logexp(c1), dt)
                W.append(logexp(c1))
                C.append(f1)
                optw = logexp(c1)
                y = yh1 / np.sum(yh1 * dt)
            else:
                a = c1
                c1 = c2
                c2 = (2 - phi) * a + (phi - 1) * b
                f1 = f2
                f2, yh2 = cost_function(y_hist,N,logexp(c2),dt)
                W.append(logexp(c2))
                C.append(f2)
                optw = logexp(c2)
                y = yh2 / np.sum(yh2 * dt)
            
            k = k + 1;
        """
        nbs = 1e3        # number of bootstrap samples
        yb = np.zeros((nbs,len(t_in)))
        
        for i in range(int(nbs)):
            idx = np.asarray(np.ceil(np.random.rand(N)*N), dtype=np.int)
            xb = x_ab[idx] 
            edges = np.hstack((t-dt/2, 10.))
            y_histb, _ = np.histogram(xb, edges)
            y_histb = y_histb/dt/N
            yb_buf = fftkernel(y_histb, optw/dt)
            yb_buf = yb_buf / np.sum(yb_buf*dt)
            interp = interp1d(t, yb_buf)
            yb[i,:] = interp(t_in)
            ybsort = np.sort(yb)
            y95b = ybsort[np.floor(0.05*nbs),:]
            y95u = ybsort[np.floor(0.95*nbs),:]
            confb95 = np.vstack((y95b, y95u))
        """

    interp = interp1d(t[:-1],y)
    y = interp(t_in[t_in<np.max(t[:-1])])
    t = t_in
    return y, t, optw


def load_data(filename):
    f = nix.File.open(filename, nix.FileMode.ReadOnly)
    b = f.blocks[0]
    stim_tag = b.tags["stimulus_strong"]
    data = stim_tag.retrieve_data(0)[:]
    sampling_interval = stim_tag.references[0].dimensions[0].sampling_interval
    time = np.asarray(stim_tag.references[0].dimensions[0].axis(data.shape[0]))
    stimulus = stim_tag.retrieve_feature_data(0)[:]
    f.close()
    return data, stimulus, sampling_interval, time

if __name__ == "__main__":
    import nix
    data, stim, sampling_interval, time = load_data('../data/data_ampullary.h5')
    spike_indices,_ = np.nonzero(data)
    spike_times = time[spike_indices]
    y,t, w = sskernel(spike_times)
    embed()



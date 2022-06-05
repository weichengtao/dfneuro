import numpy as np
from scipy import signal

def interpolation(lfp: np.ndarray, spikes: list[np.ndarray], onset: int | float, duration: int | float, u: int = 30, copy: bool = False) -> np.ndarray:
    '''
    Input:
        lfp:
            shape: ( m epochs, n samples )
            fs: u * 1,000 Hz
            unit: miuV
            align_to: epoch onset
        spikes:
            shape: m epochs of ( >= 0 spikes, )
            fs: spike
            unit: ms
            precision: u * 1,000 Hz
            align_to: epoch onset
        onset:
            unit: ms
            align_to: spike onset
        duration:
            unit: ms
        u = 30:
            unit: sample / ms
        copy = False:
            True -> copy lfp before interpolation
            False -> perform interpolation in place
    
    Output:
        interpolated_lfp:
            shape: m epochs * n samples
            fs: u * 1,000 Hz
            unit: miuV
            align_to: epoch onset

    Notice:
        1) onset usually is a negative number
        2) if (spike + onset <= 0) or (spike + onset + duration >= n / u), the spike will not be interpolated

    Example:
        1) interpolated_lfp = compute.interpolation(lfp, spikes, -0.2, 1.5, copy=True)
    '''
    if copy:
        lfp = lfp.copy()
    m, n = lfp.shape
    remove_edge = lambda s: s[(s + onset > 0) & (s + onset + duration < n / u)]
    spikes = [np.rint(remove_edge(epoch_spikes) * u).astype(int) for epoch_spikes in spikes]
    onset = int(np.rint(onset * u))
    duration = int(np.rint(duration * u))
    for i in range(m):
        epoch_lfp = lfp[i]
        epoch_spikes = spikes[i]
        for j in range(onset, onset + duration):
            epoch_lfp[epoch_spikes + j] = np.nan
        nans = np.isnan(epoch_lfp)
        epoch_lfp[nans] = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], epoch_lfp[~nans])
        lfp[i] = epoch_lfp
    return lfp

def iir(btype: str, band: list[int | float] | int | float, order: int, sig: np.ndarray, fs: int) -> np.ndarray:
    sos = signal.butter(order, band, btype, output='sos', fs=fs)
    res = signal.sosfiltfilt(sos, sig)
    return res

def downsampling(sig: np.ndarray, factor: int = 30, ftype: str = 'fir') -> np.ndarray:
    return signal.decimate(sig, factor, ftype=ftype)

def multitaper_spectrogram(lfp: np.ndarray, fmin: int, fmax: int, window_width: int | float, half_bandwidth: int | float, fs: int = 1000, n_jobs: int = 1) -> tuple[np.ndarray, ...]:
    '''
    Input:
        lfp:
            shape: ( m epochs, n samples )
            fs: fs
            unit: miuV
            align_to: epoch onset
        fmin, fmax:
            20, 120 Hz -> 20-120 Hz of the spectrogram will be returned
        window_width:
            unit: second
        half_bandwidth:
            unit: Hz
        fs = 1000:
            30,000 Hz -> raw lfp
            1,000 Hz -> downsampled lfp

    Output:
        f:
            align_to: 0 Hz
        t:
            align_to: epoch onset
        Sxx:
            shape: ( m epochs, fmax - fmin + 1 frequencies, n samples / fs * 1000 + 1)
            fs: 1 Hz * 1 ms
            unit: miuV^2/Hz
            align_to: epoch onset
    '''
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
    M = int(fs * window_width)
    NW = window_width * half_bandwidth
    K = int(np.floor(NW * 2) - 1)
    dpss_windows = signal.windows.dpss(M, NW, K, sym=True, norm=2, return_ratios=False)
    noverlap = int(M - fs / 1000)
    fmax_exclusive = fmax + 1
    res = []
    for i in range(len(lfp)):
        epoch_lfp = lfp[i]
        if Parallel:
            get_Sxx = lambda window: signal.spectrogram(epoch_lfp, fs, window, M, noverlap, fs)[-1][fmin:fmax_exclusive]
            epoch_Sxx = Parallel(n_jobs=n_jobs, verbose=0)(delayed(get_Sxx)(window) for window in dpss_windows)
        else:
            epoch_Sxx = []
            for window in dpss_windows:
                Sxx = signal.spectrogram(epoch_lfp, fs, window, M, noverlap, fs)[-1][fmin:fmax_exclusive]
                epoch_Sxx.append(Sxx)
        res.append(np.asarray(epoch_Sxx).mean(axis=0))
    f, t = signal.spectrogram(lfp[0], fs, dpss_windows[0], M, noverlap, fs)[:-1]
    f = f[fmin:fmax_exclusive]
    return f, t, np.asarray(res)

def burst(sig: np.ndarray, wmin: int | float) -> tuple[list[tuple[int, int]], float]:
    m = sig.mean()
    sd = sig.std()
    thresh = m + 2 * sd
    above_thresh = np.nonzero(sig > thresh)[0]
    diff = np.diff(above_thresh)
    left = np.nonzero(diff > 1)[0] + 1
    left = np.insert(left, 0, 0)
    left = np.append(left, len(above_thresh))
    width = np.diff(left)
    res = []
    for i, w in enumerate(width):
        if w > wmin:
            res.append((above_thresh[left[i]], above_thresh[left[i + 1] - 1]))
    return res, thresh

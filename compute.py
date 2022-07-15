from typing import Callable
import numpy as np
from scipy import signal, stats
from numba import njit
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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

def firing_rate(spikes: np.ndarray, duration: int | float, fs: int = 1000, kernel: np.ndarray | None = None) -> np.ndarray:
    firing_rate = np.zeros(int(duration * fs))
    rint_spikes = np.rint(spikes / 1000 * fs).astype(int)
    firing_rate[rint_spikes] = 1
    if not kernel:
        # boxcar kernel, 50ms duration, normalized by duration
        kernel_duration = 0.05
        kernel_width = int(kernel_duration * fs)
        kernel = np.ones(kernel_width) / kernel_duration
    firing_rate = signal.convolve(firing_rate, kernel, mode='same')
    return firing_rate

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

def is_gamma_mod(f: np.ndarray, t: np.ndarray, Sxx: np.ndarray, pre_duration: int | float = 0.4) -> bool:
    near = lambda arr, x: np.argmin((arr - x) ** 2) # find nearest index of x in arr
    t_ = t - pre_duration # t realigned to stimulus onset
    gmin, gmax = 50, 120
    pre = Sxx[:, near(f, gmin):near(f, gmax) + 1, near(t_, -0.2):near(t_, 0)].mean(axis=(1, 2))
    post = Sxx[:, near(f, gmin):near(f, gmax) + 1, near(t_, 0.1):near(t_, 0.3)].mean(axis=(1, 2))
    _, p = stats.wilcoxon(pre, post, alternative='less')
    return p < 0.05

def burst(sig: np.ndarray, wmin: int | float, thresh: int | float | None = None, greater: bool = True) -> tuple[list[tuple[int, int]], float]:
    '''
    Input:
        sig:
            fs: 1,000 Hz
            unit: miuV
        wmin:
            unit: ms
        thresh: optional
            default: mean + 2 * sd of sig
            unit: miuV
        greater:
            True -> greater than thresh
            False -> less than thresh
    Output:
        bur: list[tuple[start_idx, end_idx]]
            unit: ms
            align_to: signal onset
        thresh: threshold of magnitude for extracting burst (computed based on sig)
    '''
    if not thresh:
        m = sig.mean()
        sd = sig.std()
        thresh = m + 2 * sd
    if greater:
        above_thresh = np.nonzero(sig > thresh)[0]
    else:
        above_thresh = np.nonzero(sig < thresh)[0]
    diff = np.diff(above_thresh)
    left = np.nonzero(diff > 1)[0] + 1
    left = np.insert(left, 0, 0)
    left = np.append(left, len(above_thresh))
    width = np.diff(left)
    res = []
    for i, w in enumerate(width):
        if w > wmin:
            res.append((int(above_thresh[left[i]]), int(above_thresh[left[i + 1] - 1])))
    return res, thresh

def combine_burst(burst_list: list[list[tuple[int, int]]], epoch_samples: int, wmin: int | float = 0, overlap: bool = False, offset_samples: int = 0) -> list[tuple[int, int]]:
    if (overlap or offset_samples > 0) and wmin == 0:
        raise ValueError('if overlap == True or offset_samples > 0, the wmin should be greater than 0 to prevent ZeroDivisionError in downstream processing')
    sig = np.zeros(epoch_samples)
    for bur in burst_list:
        for start, end in bur:
            sig[start:end + 1] = sig[start:end + 1] + 1
    if offset_samples > 0:
        sig = sig[offset_samples:]
    if overlap:
        res = burst(sig, wmin, len(burst_list) - 0.5)[0]
    else:
        res = burst(sig, wmin, 0.5)[0]
    return res

def state_duration(bursts: list[tuple[int, int]]) -> int:
    return int(np.sum([b[1] - b[0] + 1 for b in bursts]))

@njit
def state_duration_jit(bursts: np.ndarray) -> int:
    return int(np.sum(bursts[:, 1] - bursts[:, 0] + 1))

def shuffle_burst(bursts: list[tuple[int, int]], off_burst_duration: int, rng: int | np.random.RandomState | None = None) -> list[tuple[int, int]]:
    bursts = bursts[:]
    n_burst = len(bursts)
    # choose n_burst samples from total
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    slots = np.sort(rng.choice(off_burst_duration + 1, n_burst, replace=False))
    # shuffle the order of bursts
    rng.shuffle(bursts)
    # insert burst before those samples
    res = []
    offset = 0
    for i, b in enumerate(bursts):
        s = int(slots[i])
        b_duration = b[1] - b[0] + 1
        res.append((offset + s, offset + s + b_duration - 1))
        offset += b_duration
    return res

@njit
def shuffle_burst_jit(bursts: np.ndarray, off_burst_duration: int) -> list[tuple[int, int]]:
    slots = np.random.choice(off_burst_duration + 1, len(bursts), replace=False)
    order = np.argsort(slots)
    res = []
    offset = 0
    for i in order:
        s = slots[i]
        b = bursts[i]
        b_duration = b[1] - b[0] + 1
        res.append((offset + s, offset + s + b_duration - 1))
        offset += b_duration
    return res

def concat_burst(bursts: list[list[tuple[int, int]]], poststim_samples: int, offset: int = 0) -> np.ndarray:
    '''
    offset:
        -x -> shift bursts to the left
        x -> shift bursts to the right
        if a burst is shifted out of the trial, it will be truncated
    '''
    n_trial = len(bursts)
    res = np.zeros((n_trial, poststim_samples))
    for i, trial_bursts in enumerate(bursts):
        for start, end in trial_bursts:
            s = min(poststim_samples, max(0, start + offset))
            e = min(poststim_samples, max(0, end + offset + 1))
            res[i, s:e] = 1
    return res.flatten()

def active_silent(Sxx: np.ndarray, bands: list[tuple[int | float, int | float]], active_sd: int | float, silent_sd: int | float, 
    i_trial: int, offset_samples: int = 0, return_duration: bool = False) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | tuple[int, int]:
    # active state
    burst_list = []
    # silent state
    burst_list_ = []
    for fmin, fmax in bands:
        wmin = 1000 / ((fmax + fmin) / 2) * 3
        sig = Sxx[i_trial, fmin-20:fmax-20 + 1].mean(axis=0)
        sig_mean = sig.mean()
        sig_sd = sig.std()
        # active state
        bur = burst(sig, wmin, thresh=sig_mean + active_sd * sig_sd)[0]
        burst_list.append(bur)
        # silent state
        bur_ = burst(sig, wmin, thresh=sig_mean + silent_sd * sig_sd, greater=False)[0]
        burst_list_.append(bur_)
    active = combine_burst(burst_list, len(sig), wmin=wmin, overlap=False, offset_samples=offset_samples)
    silent = combine_burst(burst_list_, len(sig), wmin=wmin, overlap=True, offset_samples=offset_samples)
    if return_duration:
        return state_duration(active), state_duration(silent) # in unit of ms
    return active, silent

def pev(samples, tags, conditions) -> float | None:
    samples = np.asarray(samples)
    tags = np.asarray(tags)
    grouped_samples = [samples[tags == cond] for cond in conditions]
    sst = np.sum((samples - samples.mean()) ** 2)
    if sst == 0:
        return None
    sse = np.sum([np.sum((arr - arr.mean()) ** 2) for arr in grouped_samples])
    ssb = sst - sse
    dfe = len(samples) - len(conditions)
    dfb = len(conditions) - 1
    mse = sse / dfe
    omega_squared = (ssb - dfb * mse) / (mse + sst)
    return omega_squared

@ignore_warnings(category=ConvergenceWarning)
def acc(samples, tags, conditions, n_splits: int = 4, n_repeats: int = 50, n_jobs: int = 1, rng: int | np.random.RandomState | None = None) -> np.ndarray:
    X = np.asarray(samples)[:, np.newaxis]
    le = LabelEncoder()
    le.fit(conditions)
    y = le.transform(tags)
    if isinstance(rng, int):
        rng = np.random.RandomState(rng) # splits are different across repeats
    clf = make_pipeline(StandardScaler(), LinearSVC(dual=False)) # prefer dual=False when n_samples > n_features
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=rng)
    res = []
    for i in range(n_repeats):
        scores = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
        res.extend(scores)
    return np.asarray(res)

def burst_info(bursts: list[list[tuple[int, int]]], spikes: list[np.ndarray], tags: np.ndarray, ifunc: Callable, rng: int | np.random.RandomState | None = None) -> np.ndarray | float | None:
    conditions = np.sort(np.unique(tags))
    fr_per_burst = []
    tag_per_burst = []
    for i_trial, bur in enumerate(bursts):
        trial_spikes = spikes[i_trial]
        for start, end in bur:
            burst_spikes = trial_spikes[(trial_spikes >= start) & (trial_spikes <= end)]
            fr_per_burst.append(len(burst_spikes) / (end - start)) # in unit of spikes/ms
            tag_per_burst.append(tags[i_trial])
    if rng is not None:
        return ifunc(fr_per_burst, tag_per_burst, conditions, rng=rng)
    else:
        return ifunc(fr_per_burst, tag_per_burst, conditions)

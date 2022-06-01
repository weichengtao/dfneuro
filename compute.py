import numpy as np

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

def highpass():
    pass

def lowpass():
    pass

def bandpass():
    pass

def bandstop():
    pass

def downsampling():
    pass

def spectrogram():
    pass

def gamma_burst():
    pass

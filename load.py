import os
import numpy as np
import pandas as pd
from scipy import io
import h5py
from pyns import NSFile

def find(dir: str, search_for: str) -> list[str]:
    '''
    find dir -name search_for | sort
    '''
    path = []
    for root, dirs, files in os.walk(dir):
        for name in dirs + files:
            if search_for in name:
                path.append(os.path.join(root, name))
    return sorted(path)

def events(path: str, session: int) -> tuple[pd.DataFrame, float]:
    '''
    Input:
        path: path to event_markers.csv
            unit: second
            precision: 30,000 Hz
            align_to: recording onset
        session: session to be extracted

    Output:
        events:
            unit: second
            precision: 30,000 Hz
            align_to: recording onset
    '''
    df = pd.read_csv(path)
    marks = {
        'session_on': 11000000 + int(bin(session)[2:]),
        'manual_reward_on': 1100,
        'manual_reward_off': 1000,
        'trial_start': 10, # -8
        'fix_on': 1, # -7
        'target_on': [10100000, 10100001, 10100010, 10100011], # -6 or -4
        'distractor_on': [1100000, 1100001, 1100010, 1100011], # -4
        'response_on': 101, # -2
        'left_fix': 1101, # -1
        'reward_on': 110, # 0
        'reward_off': 100, # 1
        'trial_end': 100000, # 2
    }
    res = {
        'trial_onset': [], # in unit of sec with 30000 Hz srate
        'fix_onset': [],
        'stim_0_onset': [],
        'stim_0_type': 1,
        'stim_0_loc': [], # bottom left => 0, bottom right => 1, top left => 2, top right => 3
        'stim_1_onset': [],
        'stim_1_type': [], # target => 1 or distractor => 0
        'stim_1_loc': [],
    }
    session_on = False
    for i, row in df.iterrows():
        if not session_on:
            if row['words'] == marks['session_on']:
                session_on = True
                session_onset = row['timestamps']
            else:
                continue
        if row['words'] == marks['reward_on'] and df.loc[i - 8, 'words'] == marks['trial_start']:
            if df.loc[i - 6, 'words'] not in marks['target_on']:
                raise ValueError(f'target_on code {df.loc[i - 4, "words"]} cannot be recognized')
            res['trial_onset'].append(df.loc[i - 8, 'timestamps'])
            res['fix_onset'].append(df.loc[i - 7, 'timestamps'])
            res['stim_0_onset'].append(df.loc[i - 6, 'timestamps'])
            res['stim_0_loc'].append(int(str(df.loc[i - 6, 'words'])[-2:], 2))
            res['stim_1_onset'].append(df.loc[i - 4, 'timestamps'])
            res['stim_1_type'].append(len(str(df.loc[i - 4, 'words'])) - 7)
            res['stim_1_loc'].append(int(str(df.loc[i - 4, 'words'])[-2:], 2))
    return pd.DataFrame(res), session_onset

def spiketrain(path: list[str], epoch_onsets: np.ndarray | list | tuple | int | float, ms_per_epoch: int | float) -> list[np.ndarray]:
    '''
    Input:
        path: list of paths to unit.mat
            shape: ( x spikes, )
            fs: spike
            unit: ms
            precision: 30,000 Hz
            align_to: session onset
        epoch_onsets:
            shape: ( m epochs, )
            fs: epoch
            unit: ms
            precision: 30,000 Hz
            align_to: session onset
        ms_per_epoch: epoch duration in unit of ms

    Output:
        spikes:
            shape: m epochs of ( >= 0 spikes, )
            fs: spike
            unit: ms
            precision: 30,000 Hz
            align_to: epoch onset
    '''
    if isinstance(epoch_onsets, (int, float)):
        epoch_onsets = [epoch_onsets]
    epoch_onsets = np.asarray(epoch_onsets).flatten()
    units = []
    for p in path:
        if h5py.is_hdf5(p):
            with h5py.File(p, 'r') as f:
                unit = f['timestamps'][:].flatten()
        else:
            unit = io.loadmat(p)['timestamps'].flatten()
        units.append(unit)
    mua = np.concatenate(units)
    mua.sort()
    n_epoch = len(epoch_onsets)
    n_spike = len(mua)
    i_spike = 0
    res = []
    for i in range(n_epoch):
        spikes = []
        left = epoch_onsets[i]
        right = epoch_onsets[i] + ms_per_epoch - 0.5
        for j in range(i_spike, n_spike):
            s = mua[j]
            if s < left:
                continue
            if s < right:
                spikes.append(s - left)
            else:
                i_spike = j
                break
        res.append(np.asarray(spikes))
    return res

def rawlfp(path: str, channel: int, epoch_onsets: np.ndarray | list | tuple | int | float, samples_per_epoch: int | float, n_jobs: int = 1) -> np.ndarray:
    '''
    Input:
        path: path to ns5 file
            shape: ( x samples, )
            fs: 30,000 Hz
            unit: miuV
            align_to: recording onset
        channel: channel to be extracted
        epoch_onsets:
            shape: ( m epochs, )
            fs: epoch
            unit: sample:
            precision: 30,000 Hz
            align_to: recording onset
        samples_per_epoch: epoch duration in unit of sample
        n_jobs = 1: number of jobs to be used if joblib is available

    Output:
        lfp:
            shape: ( n epochs, n samples)
            fs: 30,000 Hz
            unit: miuV
            align_to: epoch onset
    '''
    if isinstance(epoch_onsets, (int, float)):
        epoch_onsets = [epoch_onsets]
    epoch_onsets = np.asarray(epoch_onsets).flatten().astype(int)
    samples_per_epoch = int(samples_per_epoch)
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
    if Parallel:
        get_data = lambda onset: NSFile(path, proc_single=True).entities[channel - 1].get_analog_data(onset, samples_per_epoch)
        lfp = Parallel(n_jobs=n_jobs, verbose=0)(delayed(get_data)(onset) for onset in epoch_onsets)
    else:
        entity = NSFile(path, proc_single=True).entities[channel - 1]
        lfp = [entity.get_analog_data(onset, samples_per_epoch) for onset in epoch_onsets]
    return np.asarray(lfp)

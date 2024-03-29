import os, pickle
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

def _strobe_to_words(strobes):
    words = []
    if (strobes[0] == 4415) or (strobes[0] == 4606) or (strobes[0] == 4159): 
        for sv in strobes:
            w = np.binary_repr(~sv, 16)[-8:]
            words.append(w)
    elif strobes[0] == -4416:
        for sv in strobes:
            w = np.binary_repr(2**15-abs(sv), 16)[-8:]
            words.append(w)
    elif strobes[0] == -64:
        for sv in strobes:
            if sv < 0:
                w = np.binary_repr(2**16 - abs(sv), 16)[-8:]
            else:
                w = np.binary_repr(sv, 16)[-8:]
            words.append(w)
    elif (strobes.min() == 63) or (strobes.min() == 77) or (strobes.min() == 2069) or (strobes.min() == 2071) or (strobes.min() == 2271):
        for sv in strobes:
            w = np.binary_repr(~sv, 16)[-8:]
            words.append(w)
    else:
        for sv in strobes:
            w = np.binary_repr(w, 16)[-8:]
            words.append(w)
    return words

def _mark_to_loc(mark, is_old_task: bool = False):
    if is_old_task:
        row = int(mark[7:4:-1], 2) - 2
        col = int(mark[4:1:-1], 2) - 2
    else:
        i = int(str(mark)[-2:], 2)
        if i == 0:
            row, col = 2, 0
        elif i == 1:
            row, col = 2, 2
        elif i == 2:
            row, col = 0, 0
        elif i == 3:
            row, col = 0, 2
    return f'r{row}_c{col}'

def events(path: str, session: int = 1, is_old_task: bool = False, allow_single_stim: bool = False, allow_code_mismatch: bool = False) -> tuple[pd.DataFrame, float]:
    '''
    Input:
        path:
            path to event_markers.csv
            unit: second
            precision: 30,000 Hz
            align_to: recording onset
            or path to event_data.mat
            unit: second
            precision: 40,000 Hz
            align_to: recording onset
        session: session to be extracted

    Output:
        events:
            unit: second
            precision: 30,000 Hz
            align_to: recording onset
            or
            unit: second
            precision: 40,000 Hz
            align_to: recording onset
    '''
    if is_old_task:
        marks = {
            "session_on": "11000000",
            "trial_start": "00000000",
            "fix_on": "00000001",
            "target_on": [
                '01010010', '01110010', '01001010',
                '01010110',             '01001110',
                '01010001', '01110001', '01001001',
            ],
            "distractor_on": [
                '10010010', '10110010', '10001010',
                '10010110',             '10001110',
                '10010001', '10110001', '10001001',
            ],
            "delay_1_on": "00000011",
            "delay_2_on": "00000100",
            "response_on": "00000101",
            "reward_on": "00000110",
            "manual_reward_on": "00001000",
            "failure": "00000111",
            "trial_end": "00100000"
        }
        res = {
            'trial_onset': [], # in unit of sec with 40000 Hz srate
            'fix_onset': [],
            'stim_1_onset': [],
            'stim_1_type': 1, # target => 1 or distractor => 0
            'stim_1_loc': [],
            'stim_2_onset': [],
            'stim_2_type': 0,
            'stim_2_loc': [],
        }
        with h5py.File(path, 'r') as f:
            strobes = f['sv'][:].flatten().astype(int)
            timestamps = f['ts'][:].flatten()
        df = pd.DataFrame({
            'words': _strobe_to_words(strobes),
            'timestamps': timestamps
        })
        session_on = False
        for i in range(len(df)):
            if not session_on:
                if df.loc[i, 'words'] == marks['session_on']:
                    session_on = True
                    session_onset = df.loc[i, 'timestamps']
                else:
                    continue
            if (df.loc[i, 'words'] == marks['reward_on'] and
                df.loc[i - 1, 'words'] == marks['response_on'] and
                df.loc[i - 6, 'words'] == marks['fix_on'] and
                df.loc[i - 7, 'words'] == marks['trial_start']):
                t_stim2 = df.loc[i - 3, 'timestamps']
                t_delay1 = df.loc[i - 4, 'timestamps']
                t_stim1 = df.loc[i - 5, 'timestamps']
                dt_1 = t_delay1 - t_stim1 # should be around 0.3
                dt_2 = t_stim2 - t_delay1 # should be around 1
                if dt_1 > 0.35 or dt_1 < 0.25 or dt_2 > 1.05 or dt_2 < 0.95:
                    continue
                res['trial_onset'].append(df.loc[i - 7, 'timestamps'])
                res['fix_onset'].append(df.loc[i - 6, 'timestamps'])
                res['stim_1_onset'].append(t_stim1)
                res['stim_1_loc'].append(_mark_to_loc(df.loc[i - 5, 'words'], True))
                res['stim_2_onset'].append(t_stim2)
                res['stim_2_loc'].append(_mark_to_loc(df.loc[i - 3, 'words'], True))
        return pd.DataFrame(res), 0
    df = pd.read_csv(path)
    marks = {
        'session_on': 11000000 + int(bin(session)[2:]),
        'next_session_on': 11000000 + int(bin(session + 1)[2:]),
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
        'stim_1_onset': [],
        'stim_1_type': 1,
        'stim_1_loc': [], # bottom left => 0, bottom right => 1, top left => 2, top right => 3
        'stim_2_onset': [],
        'stim_2_type': [], # target => 1 or distractor => 0
        'stim_2_loc': [],
    }
    session_on = False
    for i, row in df.iterrows():
        if not session_on:
            if row['words'] == marks['session_on']:
                session_on = True
                session_onset = row['timestamps']
            else:
                continue
        if session_on and row['words'] == marks['next_session_on']:
            break
        if row['words'] == marks['reward_on'] and (df.loc[i - 8, 'words'] == marks['trial_start'] or df.loc[i - 6, 'words'] == marks['trial_start']):
            if df.loc[i - 6, 'words'] in marks['target_on']:
                res['trial_onset'].append(df.loc[i - 8, 'timestamps'])
                res['fix_onset'].append(df.loc[i - 7, 'timestamps'])
                res['stim_1_onset'].append(df.loc[i - 6, 'timestamps'])
                res['stim_1_loc'].append(_mark_to_loc(df.loc[i - 6, 'words']))
                res['stim_2_onset'].append(df.loc[i - 4, 'timestamps'])
                res['stim_2_type'].append(len(str(df.loc[i - 4, 'words'])) - 7)
                res['stim_2_loc'].append(_mark_to_loc(df.loc[i - 4, 'words']))
            elif allow_single_stim and df.loc[i - 4, 'words'] in marks['target_on']:
                res['trial_onset'].append(df.loc[i - 6, 'timestamps'])
                res['fix_onset'].append(df.loc[i - 5, 'timestamps'])
                res['stim_1_onset'].append(df.loc[i - 4, 'timestamps'])
                res['stim_1_loc'].append(_mark_to_loc(df.loc[i - 4, 'words']))
                res['stim_2_onset'].append(None)
                res['stim_2_type'].append(None)
                res['stim_2_loc'].append(None)
            elif not allow_code_mismatch:
                raise ValueError(f'target_on code {df.loc[i - 6, "words"]} at {i - 6} {df.loc[i - 6, "timestamps"]:.3f} cannot be recognized')
    return pd.DataFrame(res), session_onset

def spiketrain(path: list[str] | str, epoch_onsets: np.ndarray | list | tuple | int | float, ms_per_epoch: int | float) -> list[np.ndarray]:
    '''
    Input:
        path: list of paths or a single path to unit.mat
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
    if isinstance(path, str):
        path = [path]
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

def lplfp(path: str, epoch_onsets: np.ndarray | list | tuple | int | float, samples_per_epoch: int | float, is_pickle: bool = False) -> np.ndarray:
    if isinstance(epoch_onsets, (int, float)):
        epoch_onsets = [epoch_onsets]
    epoch_onsets = np.rint(np.asarray(epoch_onsets)).flatten().astype(int)
    samples_per_epoch = int(samples_per_epoch)
    if is_pickle:
        with open(path, 'rb') as f:
            lfp_data = pickle.load(f)
    else:
        with h5py.File(path, 'r') as f:
            lfp_data = f['lowpassdata/data/data'][:].flatten().astype(float)
    res = [lfp_data[onset:onset + samples_per_epoch] for onset in epoch_onsets]
    return np.asarray(res)

def rawlfp(path: str, channel: int, epoch_onsets: np.ndarray | list | tuple | int | float, samples_per_epoch: int | float, n_jobs: int = 0) -> np.ndarray:
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
    if Parallel and n_jobs > 0:
        get_data = lambda onset: NSFile(path, proc_single=True).entities[channel - 1].get_analog_data(onset, samples_per_epoch)
        lfp = Parallel(n_jobs=n_jobs, verbose=0)(delayed(get_data)(onset) for onset in epoch_onsets)
    else:
        entity = NSFile(path, proc_single=True).entities[channel - 1]
        lfp = [entity.get_analog_data(onset, samples_per_epoch) for onset in epoch_onsets]
    return np.asarray(lfp)

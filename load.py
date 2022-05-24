import pandas as pd

def events(df: pd.DataFrame, session: int) -> pd.DataFrame:
    '''
    unit: second
    srate: 30,000 Hz
    align_to: recording onset
    '''
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
        'session_onset': 0, # in unit of sec with 30000 Hz srate
        'trial_onset': [],
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
                res['session_onset'] = row['timestamps']
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
    return pd.DataFrame(res)

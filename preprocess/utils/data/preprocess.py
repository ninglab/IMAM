import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import pickle as pkl
import pdb

def get_session_id(df, interval):
    df_prev = df.shift()
    is_new_session = (df.userId != df_prev.userId) | (
        df.timestamp - df_prev.timestamp > interval
    )
    session_id = is_new_session.cumsum() - 1
    return session_id


def group_sessions(df, interval):
    sessionId = get_session_id(df, interval)
    df = df.assign(sessionId=sessionId)
    return df


def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('sessionId', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    return df_long


def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    return df_freq


def filter_until_all_long_and_freq(df, min_len=2, min_support=5):
    while True:
        df_long = filter_short_sessions(df, min_len)
        df_freq = filter_infreq_items(df_long, min_support)
        if len(df_freq) == len(df):
            break
        df = df_freq
    return df


def truncate_long_sessions(df, max_len=20, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['sessionId', 'timestamp'])
    itemIdx = df.groupby('sessionId').cumcount()
    df_t = df[itemIdx < max_len]
    return df_t


def update_id(df, field):
    labels, uniques = pd.factorize(df[field])
    kwargs = {field: labels}
    if field == 'itemId':
        oid2aid = {oid:aid for oid, aid in enumerate(uniques)}
        with open('oid2aid.pkl', 'wb') as f:
            pkl.dump(oid2aid, f)
    df = df.assign(**kwargs)
    return df


def remove_immediate_repeats(df):
    df_prev = df.shift()
    is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
    df_no_repeat = df[is_not_repeat]
    return df_no_repeat


def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.sessionId, df_endtime.index))
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new)
    df = df.sort_values(['sessionId', 'timestamp'])
    return df


def keep_top_n_items(df, n):
    item_support = df.groupby('itemId', sort=False).size()
    top_items = item_support.nlargest(n).index
    df_top = df[df.itemId.isin(top_items)]
    return df_top


def split_by_time(df, timedelta, yoochoose=False):
    max_time = df.timestamp.max()
    end_time = df.groupby('sessionId').timestamp.max()
    split_time = max_time - timedelta
    train_sids = end_time[end_time < split_time].index
    if yoochoose:
        end_time_train = end_time[end_time.index.isin(train_sids)]
        end_time_train = end_time_train.sort_values()
        cutoff_1_64 = len(end_time_train)//64
        cutoff_1_4 = len(end_time_train)//4
        train_sids_1_64 = end_time_train.index[-cutoff_1_64:]
        train_sids_1_4 = end_time_train.index[-cutoff_1_4:]
        df_test = df[~df.sessionId.isin(train_sids)]
        df_train_1_64 = df[df.sessionId.isin(train_sids_1_64)]
        df_train_1_4  = df[df.sessionId.isin(train_sids_1_4)]
        return df_train_1_4, df_train_1_64, df_test
    else:
        df_train = df[df.sessionId.isin(train_sids)]
        df_test = df[~df.sessionId.isin(train_sids)]
        return df_train, df_test


def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test

def valid_split(df, valid_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_valid = int(len(endtime) * valid_split)
    valid_session_ids = endtime.index[-num_valid:]
    df_train_valid = df[~df.sessionId.isin(valid_session_ids)]
    df_test_valid = df[df.sessionId.isin(valid_session_ids)]
    return df_train_valid, df_test_valid


def save_sessions(df, filepath):
    df = reorder_sessions_by_endtime(df)
    sessions = df.groupby('sessionId').itemId.apply(lambda x: ','.join(map(str, x)))
    sessions.to_csv(filepath, sep='\t', header=False, index=False)


def save_dataset(dataset_dir, df_train, df_test, yoochoose=0, valid=0):
    # filter items in test but not in train
    df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
    df_test = filter_short_sessions(df_test)

    if not valid:
        print(f'No. of Clicks: {len(df_train) + len(df_test)}')
        print(f'No. of Items: {df_train.itemId.nunique()}')
    else:
        print(f'No. of Clicks in validation: {len(df_train) + len(df_test)}')
        print(f'No. of Items in validation: {df_train.itemId.nunique()}')

    #train and validation share same index
    # update itemId
    train_itemId_new, uniques = pd.factorize(df_train.itemId)
    #shift by 1
    train_itemId_new += 1
    df_train = df_train.assign(itemId=train_itemId_new)
    #shift by 1
    oid2nid = {oid: i+1 for i, oid in enumerate(uniques)}
    test_itemId_new = df_test.itemId.map(oid2nid)
    df_test = df_test.assign(itemId=test_itemId_new)
    if not valid:
        nid2oid = {v:k for k, v in oid2nid.items()}
        with open('nid2oid.pkl', 'wb') as f:
            pkl.dump(nid2oid, f)

    num_items = len(uniques)
    if yoochoose == 1:
        dataset_dir = Path(str(dataset_dir)+'1_4')
    elif yoochoose == 2:
        dataset_dir = Path(str(dataset_dir)+'1_64')
    print(f'saving dataset to {dataset_dir}')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not valid:
        save_sessions(df_train, dataset_dir / 'train.txt')
        save_sessions(df_test, dataset_dir / 'test.txt')
        with open(dataset_dir / 'num_items.txt', 'w') as f:
            f.write(str(num_items))
    else:
        save_sessions(df_train, dataset_dir / 'train_valid.txt')
        save_sessions(df_test, dataset_dir / 'test_valid.txt')
        with open(dataset_dir / 'num_items_valid.txt', 'w') as f:
            f.write(str(num_items))


def preprocess_diginetica(dataset_dir, csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 2, 3, 4],
        delimiter=';',
        parse_dates=['eventdate'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    # timeframe (time since the first query in a session, in milliseconds)
    df['timestamp'] = pd.to_timedelta(df.timeframe, unit='ms') + df.eventdate
    df = df.drop(['eventdate', 'timeframe'], 1)
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = filter_infreq_items(df)
    df = filter_short_sessions(df)
    df_train, df_test = split_by_time(df, pd.Timedelta(days=7))
    df_train_valid, df_test_valid = valid_split(df_train)
    save_dataset(dataset_dir, df_train, df_test)
    save_dataset(dataset_dir, df_train_valid, df_test_valid, valid=1)

def preprocess_yoochoose(dataset_dir, csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 1, 2, 3],
        delimiter=',',
        parse_dates=['timestamp'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    df['timestamp'] = df.timestamp.apply(lambda x: x.timestamp())
    df = df.drop(['categoryId'], 1)
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)
    df = filter_infreq_items(df)
    df = filter_short_sessions(df)
    df_train_1_4, df_train_1_64, df_test = split_by_time(df, 86400, yoochoose=True)
    df_train_1_4_valid, df_test_1_4_valid = valid_split(df_train_1_4)
    df_train_1_64_valid, df_test_1_64_valid = valid_split(df_train_1_64)
    save_dataset(dataset_dir, df_train_1_4, df_test, yoochoose=1)
    save_dataset(dataset_dir, df_train_1_64, df_test, yoochoose=2)
    save_dataset(dataset_dir, df_train_1_4_valid, df_test_1_4_valid, yoochoose=1, valid=1)
    save_dataset(dataset_dir, df_train_1_64_valid, df_test_1_64_valid, yoochoose=2, valid=1)

def preprocess_gowalla_lastfm(dataset_dir, csv_file, usecols, interval, n):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        sep='\t',
        header=None,
        names=['userId', 'timestamp', 'itemId'],
        usecols=usecols,
        parse_dates=['timestamp'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    df = df.dropna()
    df = update_id(df, 'userId')
    df = update_id(df, 'itemId')
    df = df.sort_values(['userId', 'timestamp'])

    df = group_sessions(df, interval)
    df = remove_immediate_repeats(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = keep_top_n_items(df, n)
    df = filter_until_all_long_and_freq(df)
    df_train, df_test = train_test_split(df, test_split=0.2)
    df_train_valid, df_test_valid = valid_split(df_train)
    save_dataset(dataset_dir, df_train, df_test)
    save_dataset(dataset_dir, df_train_valid, df_test_valid, valid=1)

"""data_formatters


"""
import pandas as pd
import numpy as np
import FMA_utils as utils

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

class DataFormatter(object):
    def __init__(self, path):
        self.path = path

    def do_format(self):
        raise NotImplemented

    def do_nomalisation(self):
        raise NotImplemented


class FmaGenresDataFormatter(DataFormatter):
    def __init__(self, path):
        self.genres = pd.read_csv(path,  sep=',', engine='c', header=None, na_filter=False, low_memory=False).values
        super(FmaGenresDataFormatter, self).__init__(path=path)

    def do_format(self):
        pass


class FmaFeaturesDataFormatter(DataFormatter):
    def __init__(self, path):
        super(FmaFeaturesDataFormatter, self).__init__(path=path)

    def do_format(self):
        pass


def normalize_nparray(array):
    array_Mean = np.mean(array, axis=0)
    array_Std = np.std(array, axis=0)
    array = array - array_Mean
    array = array / array_Std
    return array

def cutting_datasets(data):
    train_flag = int(data.shape[0] * TRAIN_RATIO)
    valid_flag = train_flag + int(data.shape[0] * VALID_RATIO)
    train = data[:train_flag]
    valid = data[train_flag + 1:valid_flag]
    test = data[valid_flag + 1:]
    return train, valid, test

def FMA_genres(tracks, genres, features, echonest):
    # i = 4858
    corr_table = {}
    i = 0
    for index, row in genres.iterrows():
        i += 1
        corr_table[str(index)] = i

    combined_set_targets = tracks['track', 'genres'].to_frame()
    combined_set_inputs = features.values

    i += 1
    null_index = i
    for index, row in combined_set_targets.iterrows():
        temp_str = ""
        if not row[0]:
            row[0] = null_index
        else:
            for digit in row[0]:
                temp_str += str(digit)
                temp_str += ','
            temp_str = temp_str[:-1]
            if temp_str in corr_table:
                row[0] = corr_table[temp_str]
            else:
                i += 1
                corr_table[temp_str] = i
                row[0] = i

    combined_set_targets = combined_set_targets.values
    combined_set_targets = combined_set_targets.astype(np.float32)
    combined_set_inputs = normalize_nparray(combined_set_inputs)

    train_inputs, valid_inputs, test_inputs = cutting_datasets(combined_set_inputs)
    train_targets, valid_targets, test_targets = cutting_datasets(combined_set_targets)

    np.savez(Path + 'FMA_feature-genre-train.npz', inputs=train_inputs, targets=train_targets)
    np.savez(Path + 'FMA_feature-genre-valid.npz', inputs=valid_inputs, targets=valid_targets)
    np.savez(Path + 'FMA_feature-genre-test.npz', inputs=test_inputs, targets=test_targets)


def FMA_reduce_genres(tracks, genres, features, echonest):
    #i = 236
    corr_table = {}
    i = 0
    for index, row in genres.iterrows():
        i += 1
        corr_table[str(index)] = i

    combined_set_targets = tracks['track', 'genres'].to_frame()
    combined_set_inputs = features.values

    i += 1
    null_index = i
    for index, row in combined_set_targets.iterrows():
        if not row[0]:
            row[0] = null_index
        else:
            temp_str = str(row[0][0])
            if temp_str in corr_table:
                row[0] = corr_table[temp_str]
            else:
                i += 1
                corr_table[temp_str] = i
                row[0] = i

    combined_set_targets = combined_set_targets.values
    combined_set_targets = combined_set_targets.astype(np.float32)
    combined_set_inputs = normalize_nparray(combined_set_inputs)

    train_inputs, valid_inputs, test_inputs = cutting_datasets(combined_set_inputs)
    train_targets, valid_targets, test_targets = cutting_datasets(combined_set_targets)

    np.savez(Path + 'FMA_feature-genre-reduced-train.npz', inputs=train_inputs, targets=train_targets)
    np.savez(Path + 'FMA_feature-genre-reduced-valid.npz', inputs=valid_inputs, targets=valid_targets)
    np.savez(Path + 'FMA_feature-genre-reduced-test.npz', inputs=test_inputs, targets=test_targets)


def FMA_abandoned_genres(tracks, genres, features, echonest, keep_feature_number):
    #i =
    corr_table = {}
    i = 0
    for index, row in genres.iterrows():
        i += 1
        corr_table[str(index)] = i

    combined_set_targets = tracks['track', 'genres'].to_frame()
    combined_set_inputs = features.values

    i += 1
    null_index = i
    counter = {}
    for index, row in combined_set_targets.iterrows():
        if not row[0]:
            row[0] = null_index
        else:
            temp_str = str(row[0][0])
            if temp_str in corr_table:
                row[0] = corr_table[temp_str]
            else:
                i += 1
                corr_table[temp_str] = i
                row[0] = i

        if str(row[0]) not in counter:
            counter[str(row[0])] = 1
        else:
            counter[str(row[0])] += 1

    selection = []
    s = [(k, counter[k]) for k in sorted(counter, key=counter.get, reverse=True)]
    s = s[:keep_feature_number]
    i = 0
    corr_table2 = {}
    for k, v in s:
        selection.append(int(k))
        if k not in corr_table2:
            corr_table2[k] = i
            i += 1

    combined_set_targets = combined_set_targets.values
    combined_set_targets = combined_set_targets.astype(np.float32)
    combined_set_inputs = normalize_nparray(combined_set_inputs)

    drop = []
    j = 0
    for x in np.nditer(combined_set_targets):
        if x not in selection:
            drop.append(j)
        j += 1

    combined_set_targets = np.delete(combined_set_targets, drop)
    combined_set_inputs = np.delete(combined_set_inputs, drop, axis=0)

    for x in np.nditer(combined_set_targets, op_flags=['readwrite']):
        x[...] = corr_table2[str(int(x[...]))]
        i += 1

    train_inputs, valid_inputs, test_inputs = cutting_datasets(combined_set_inputs)
    train_targets, valid_targets, test_targets = cutting_datasets(combined_set_targets)

    np.savez(Path + 'FMA_feature-genre-abandoned-train.npz', inputs=train_inputs, targets=train_targets)
    np.savez(Path + 'FMA_feature-genre-abandoned-valid.npz', inputs=valid_inputs, targets=valid_targets)
    np.savez(Path + 'FMA_feature-genre-abandoned-test.npz', inputs=test_inputs, targets=test_targets)

if __name__ == "__main__":
    Path = 'data/'
    # loaded = np.load('data/FMA_feature-genre.npz')
    # inputs, targets = loaded['inputs'], loaded['targets']
    tracks = utils.load(Path + 'tracks.csv')
    genres = utils.load(Path + 'genres.csv')
    features = utils.load(Path + 'features.csv')
    echonest = utils.load(Path + 'echonest.csv')

    np.testing.assert_array_equal(features.index, tracks.index)
    assert echonest.index.isin(tracks.index).all()

    FMA_abandoned_genres(tracks, genres, features, echonest, 10)
    FMA_genres(tracks, genres, features, echonest)
    FMA_reduce_genres(tracks, genres, features, echonest)


    # test = np.load(Path + 'FMA_feature-genre.npz')
    #
    # combined_set_targets = test['targets']
    # combined_set_targets = test['inputs']
    print()


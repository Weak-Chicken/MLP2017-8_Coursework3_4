"""data_formatters


"""
import pandas as pd
import numpy as np
import utils.utils


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

    corr_table = {}
    i = 0
    for index, row in genres.iterrows():
        i += 1
        corr_table[str(i)] = index

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
    np.savez(Path + 'FMA_feature-genre.npz', inputs=combined_set_inputs, targets=combined_set_targets)

    # test = np.load(Path + 'FMA_feature-genre.npz')
    #
    # combined_set_targets = test['targets']
    # combined_set_targets = test['inputs']
    print()

# i = 4858
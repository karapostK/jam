import glob
import logging
import os

import numpy as np


class FeatureHolder:
    """
    Looks at the folder and load all .npy files as users/items features.

    File formats:
        <user/item>_<modality>_features.npy             where modality can be text, image, audio, cf etc.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

        self.user_features = None
        self.item_features = None

        self.n_user_features = None
        self.n_item_features = None

        self._load_data()

        logging.info(f'Built FeatureHolder module\n'
                     f'- data_path: {self.data_path}\n'
                     f'- n_user_features: {self.n_user_features}\n'
                     f'- n_item_features: {self.n_item_features}')

    def _load_data(self):
        logging.info('Loading data')

        # Loading user representations, if any
        self.user_features = dict()
        user_features_matching_files = glob.glob(os.path.join(self.data_path, 'user_*_features.npy'))
        for file in user_features_matching_files:
            modality = file.split('_')[1]
            self.user_features[modality] = np.load(file, allow_pickle=True)
            logging.info(f'Loaded {modality} user features. Shape is {self.user_features[modality].shape}')
        self.n_user_features = len(self.user_features)

        # Loading item representations, if any
        self.item_features = dict()
        item_features_matching_files = glob.glob(os.path.join(self.data_path, 'item_*_features.npy'))
        for file in item_features_matching_files:
            modality = file.split('_')[1]
            self.item_features[modality] = np.load(file, allow_pickle=True)
            logging.info(f'Loaded {modality} item features. Shape is {self.item_features[modality].shape}')
        self.n_item_features = len(self.item_features)

        logging.info('Finished loading data')


if __name__ == '__main__':
    fh = FeatureHolder('./amazon23office/processed')
    print(fh.user_features)
    print(fh.item_features)
    print(fh.n_user_features)
    print(fh.n_item_features)

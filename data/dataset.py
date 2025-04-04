import ast
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.data_processing import QueryProcessor

"""
File formats:
    <train/val/test>_split.tsv: query_idx \t text \t user_idx \t [item_idx_0,...,item_idx_n]        -> Header: query_idx, text, user_idx, item_idxs
    user_idxs.tsv: user_idx, <dataset_user_id>                                                      -> Header: user_idx, <dataset_user_id>
    item_idxs.tsv: item_idx, <dataset_item_id>                                                      -> Header: item_idx, <dataset_item_id>
"""


class TrainQueryDataset(Dataset):
    """
    Note on internal data:
    - triplets: DataFrame with columns query_idx, user_idx, item_idx
    - query2itemsSet: dictionary query_idx -> set(item_idxs)
    - queryData: DataFrame with columns query_idx, text
    - query2embedding: dictionary query_idx -> embedded_query
    """

    def __init__(self, data_path: str, lang_model_conf: dict):
        self.data_path = data_path

        processor = QueryProcessor(data_path, lang_model_conf, split_set='train')

        # Filled by _load_data
        self.triplets = None
        self.n_users = None
        self.n_items = None
        self.query2itemsSet = None
        self.queryData = None

        self._load_data()
        self.query2embedding = processor.process_data(self.queryData)

        logging.info(f'Built TrainQueryDataset module\n'
                     f'- data_path: {self.data_path}\n'
                     f'- n_users: {self.n_users}\n'
                     f'- n_items: {self.n_items}\n'
                     f'- n_triplets: {len(self.triplets)}')

    def _load_data(self):
        logging.info('Loading data')

        data = pd.read_csv(
            os.path.join(self.data_path, 'train_split.tsv'), sep='\t',
            converters={'item_idxs': ast.literal_eval}
        )

        # Building query2itemsSet
        self.query2itemsSet = {query_idx: set(items) for query_idx, items in zip(data['query_idx'], data['item_idxs'])}

        # Flattening data to have one item per row
        self.triplets = data[['query_idx', 'user_idx', 'item_idxs']].explode('item_idxs').rename(
            columns={'item_idxs': 'item_idx'})

        # Reading # of users and items
        self.n_users = pd.read_csv(os.path.join(self.data_path, 'user_idxs.tsv')).shape[0]
        self.n_items = pd.read_csv(os.path.join(self.data_path, 'item_idxs.tsv')).shape[0]

        # Keeping track of query data
        self.queryData = data[['query_idx', 'text']]

        logging.info('Finished loading data')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        entry = self.triplets.iloc[index]

        embedded_query = self.query2embedding[entry['query_idx']]

        return entry['query_idx'], embedded_query, entry['user_idx'], entry['item_idx']


class EvalQueryDataset(Dataset):
    """
    Note on internal data:
    - queries: DataFrame with columns query_idx, user_idx, item_idxs
    - excludeData: dictionary query_idx -> np.array(item_idxs)
    - query2embedding: dictionary query_idx -> embedded_query
    - queryData: DataFrame with columns query_idx, text
    """

    def __init__(self, data_path: str, split_set: str, lang_model_conf: dict):
        assert split_set in ['val', 'test'], f'<{split_set}> is not a valid split set!'
        self.data_path = data_path
        self.split_set = split_set

        processor = QueryProcessor(data_path, lang_model_conf, split_set)

        # Filled by _load_data
        self.queries = None
        self.n_users = None
        self.n_items = None
        self.excludeData = None
        self.queryData = None

        self._load_data()
        self.query2embedding = processor.process_data(self.queryData)

        logging.info(f'Built EvalQueryDataset module\n'
                     f'- data_path: {self.data_path}\n'
                     f'- split_set: {self.split_set}\n'
                     f'- n_users: {self.n_users}\n'
                     f'- n_items: {self.n_items}\n'
                     f'- n_queries: {len(self.queries)}')

    def _load_data(self):
        logging.info('Loading data')

        data = pd.read_csv(
            os.path.join(self.data_path, f'{self.split_set}_split.tsv'), sep='\t',
            converters={'item_idxs': ast.literal_eval}
        )

        data['item_idxs'] = data['item_idxs'].apply(lambda x: np.array(x))
        self.queries = data
        self.queryData = data[['query_idx', 'text']]

        # Reading # of users and items
        self.n_users = pd.read_csv(os.path.join(self.data_path, 'user_idxs.tsv')).shape[0]
        self.n_items = pd.read_csv(os.path.join(self.data_path, 'item_idxs.tsv')).shape[0]

        # Building excludeData
        data = pd.read_csv(
            os.path.join(self.data_path, 'train_split.tsv'), sep='\t',
            converters={'item_idxs': ast.literal_eval}
        )
        excludeData = {query_idx: set(items) for query_idx, items in zip(data['query_idx'], data['item_idxs'])}
        if self.split_set == 'test':
            # Adding validation data when performing test
            data = pd.read_csv(
                os.path.join(self.data_path, 'val_split.tsv'), sep='\t',
                converters={'item_idxs': ast.literal_eval}
            )

            # Code below updates the dictionary with the validation data, iterating over the val query_idxs
            # 1. If the query_idx is not in the dictionary, it adds it with the corresponding item_idxs (that's get({}))
            # 2. If the query_idx is already in the dictionary, it updates the item_idxs with the union of the two sets
            # NB. query_ids in excludeData but not in the validation set will remain unchanged
            excludeData.update(
                {
                    query_idx: set(items).union(excludeData.get(query_idx, {}))
                    for query_idx, items in zip(data['query_idx'], data['item_idxs'])
                }
            )

        self.excludeData = {k: np.array(list(v), dtype=np.int32) for k, v in excludeData.items()}

        logging.info('Finished loading data')

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        entry = self.queries.iloc[index]

        pos_items_mask = np.zeros(self.n_items)
        pos_items_mask[entry['item_idxs']] = 1

        exclude_items_mask = np.zeros(self.n_items, dtype=bool)
        items_excluded = self.excludeData.get(entry['query_idx'], np.array([], dtype=np.int32))
        exclude_items_mask[items_excluded] = True

        embedded_query = self.query2embedding[entry['query_idx']]

        return entry['query_idx'], embedded_query, entry['user_idx'], pos_items_mask, exclude_items_mask


def collate_fn_negative_sampling(batch: tuple, query2itemsSet: dict, n_items: int, n_negs=10):
    """
    Collate function for negative sampling. Performs uniform sampling of negative items.
    :param batch: list of tuples (query_idx, text_emb, user_idx, item_idx)
    :param query2itemsSet: dictionary query_idx -> set(item_idxs)
    :param n_items: number of items
    :param n_negs: number of negative samples
    :return: query_idx, text, user_idx, item_idx, neg_idxs
    """
    batch_size = len(batch)

    query_idxs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    text_embs = torch.stack([b[1] for b in batch])
    user_idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    item_idxs = torch.tensor([b[3] for b in batch], dtype=torch.long)

    neg_idxs = np.empty((batch_size, n_negs), dtype=np.int64)
    mask = np.ones_like(neg_idxs, dtype=bool)

    while mask.any():
        sampled = np.random.randint(0, high=n_items, size=mask.sum())
        neg_idxs[mask] = sampled

        for i, (query_idx, _, _, _) in enumerate(batch):
            pos_items = query2itemsSet[query_idx]
            for j in range(n_negs):
                if neg_idxs[i, j] not in pos_items:
                    mask[i, j] = False

    neg_idxs = torch.tensor(neg_idxs, dtype=torch.long)

    return query_idxs, text_embs, user_idxs, item_idxs, neg_idxs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting')

    data_path = './amazon23office/processed'

    lang_model_conf = {
        'tokenizer_name': 'answerdotai/ModernBERT-base',
        'model_name': 'answerdotai/ModernBERT-base',
        'device': 'cuda',
        'max_length': 100,
        'batch_size': 2000
    }

    # Train
    trainDataset = TrainQueryDataset(data_path, lang_model_conf=lang_model_conf)
    print(trainDataset[0])

    dataloader = DataLoader(
        trainDataset,
        batch_size=20,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_negative_sampling(batch, trainDataset.query2itemsSet, trainDataset.n_items,
                                                              n_negs=10)
    )

    for batch in dataloader:
        print(batch)
        break

    # Val
    valDataset = EvalQueryDataset(data_path, 'val', lang_model_conf=lang_model_conf)
    print(valDataset[0])

    dataloader = DataLoader(
        valDataset,
        batch_size=20,
        shuffle=True
    )

    for batch in dataloader:
        print(batch)
        break

    # Test

    testDataset = EvalQueryDataset(data_path, 'test', lang_model_conf=lang_model_conf)
    print(testDataset[0])

    dataloader = DataLoader(
        testDataset,
        batch_size=20,
        shuffle=True
    )

    for batch in dataloader:
        print(batch)
        break

    logging.info('End')

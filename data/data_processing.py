import logging
import math
import os

import pandas as pd
import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModel


class QueryProcessor:
    def __init__(self, data_path: str, lang_model_conf: dict, split_set: str = 'train'):
        self.data_path = data_path
        self.lang_model_conf = lang_model_conf
        self.split_set = split_set

        self.tokenizer = AutoTokenizer.from_pretrained(self.lang_model_conf['tokenizer_name'])
        self.model = AutoModel.from_pretrained(self.lang_model_conf['model_name']).to(self.lang_model_conf['device'])

        # Filled in process_data

        self.tokenized_queries = None
        self.embedded_queries = None

    def process_data(self, data: pd.DataFrame):
        """

        :param data: Should contain a 'text' and 'query_idx' column
        :return:
        """
        logging.info('Processing data')
        data = data.sort_values('query_idx')

        tokenizer_name = self.lang_model_conf['tokenizer_name'].split('/')[-1]
        model_name = self.lang_model_conf['model_name'].split('/')[-1]

        tokenized_queries_path = os.path.join(
            self.data_path,
            f'{tokenizer_name}_{self.split_set}_tokenized_queries.pt'
        )
        embedded_queries_path = os.path.join(
            self.data_path, f'{model_name}_{self.split_set}_embedded_queries.pt'
        )

        # If embedded queries already exist, skip tokenization
        if os.path.exists(embedded_queries_path):
            logging.info('Loading Embedded Queries (skipping tokenization)')
            self.embedded_queries = torch.load(embedded_queries_path, weights_only=False)
        else:
            self.tokenized_queries = self._tokenize_queries(data, tokenized_queries_path)
            self.embedded_queries = self._embed_queries(data, self.tokenized_queries, embedded_queries_path)

        query2embedding = {query_idx: embedded_query for query_idx, embedded_query in
                           zip(data['query_idx'], self.embedded_queries)}
        return query2embedding

    def _tokenize_queries(self, data, tokenized_queries_path):
        logging.info('Tokenization Step')

        if os.path.exists(tokenized_queries_path):
            logging.info('Loading Tokenized Queries')
            return torch.load(tokenized_queries_path, weights_only=False)
        else:
            logging.info('Tokenizing queries')
            tokenized_queries = self.tokenizer(
                data['text'].to_list(),
                padding='max_length',
                truncation=True,
                max_length=self.lang_model_conf['max_length'],
                return_tensors='pt'
            )
            torch.save(tokenized_queries, tokenized_queries_path)
            logging.info('New Tokenized queries saved')
        return tokenized_queries

    def _embed_queries(self, data, tokenized_queries, embedded_queries_path):
        logging.info('Embedding Step')
        if os.path.exists(embedded_queries_path):
            logging.info('Loading Embedded Queries')
            return torch.load(embedded_queries_path, weights_only=False)
        else:
            logging.info('Embedding queries')
            batch_size = self.lang_model_conf['batch_size']
            device = self.lang_model_conf['device']

            n_steps = math.ceil(len(data) / batch_size)
            embedded_queries = []
            for i in trange(n_steps):
                start = i * batch_size
                end = (i + 1) * batch_size
                with torch.no_grad():
                    output = self.model(
                        input_ids=tokenized_queries['input_ids'][start:end].to(device),
                        attention_mask=tokenized_queries['attention_mask'][start:end].to(device)
                    )
                embedded_queries.append(output.last_hidden_state.mean(dim=1).cpu())
            embedded_queries = torch.cat(embedded_queries)
            torch.save(embedded_queries, embedded_queries_path)
            logging.info('New Embedded queries saved')
        return embedded_queries

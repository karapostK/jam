import logging
import os
from abc import ABC, abstractmethod

import torch
from datasets import Dataset
from torch import nn

from data.feature import FeatureHolder


class BaseQueryMatchingModel(ABC, nn.Module):
    """
    Base class for Query Matching models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, q_idxs: torch.Tensor, q_text: tuple, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the affinity scores between (u,q) and i.
        NB. it can consider also negative sampling!

        :param q_idxs:  Query indexes. Shape is (batch_size,)
        :param q_text:  Query text. Shape is (batch_size,). Tuple of strings.
        :param u_idxs:  User indexes. Shape is (batch_size,)
        :param i_idxs: Item indexes. Shape is (batch_size,) or (batch_size, n_neg) or (batch_size, n_items)
        :return:
            preds: Predicted scores. Shape is (batch_size,) or (batch_size, neg) or (batch_size, n_items)
        """

    @abstractmethod
    def predict_all(self, q_idxs: torch.Tensor, q_text: tuple, u_idxs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the affinity scores between (u,q) and all items.

        :param q_idxs:  Query indexes. Shape is (batch_size,)
        :param q_text:  Query text. Shape is (batch_size,). Tuple of strings.
        :param u_idxs:  User indexes. Shape is (batch_size,)
        :return:
            preds: Predicted scores. Shape is (batch_size, n_items)
        """

    @abstractmethod
    def compute_loss(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor) -> dict:
        """
        Computes the loss given the predictions on positive and negative items.

        NB. dict should have ALWAYS a single entry with key 'loss'
        :param pos_preds: Positive predictions. Shape is (batch_size,)
        :param neg_preds: Negative predictions. Shape is (batch_size, n_neg) where n_neg is the number of negative samples
        :return:
            losses: Dictionary with the loss values. IT MUST HAVE AN ENTITY 'loss'
        """

    def save_model_to_path(self, path: str):
        """
        Saves the model to a specified path. Default implementation saves the state_dict of the model.
        :param path: Path to save the model
        """
        path = os.path.join(path, 'model.pth')
        torch.save(self.state_dict(), path)
        logging.info('Model Saved')

    def load_model_from_path(self, path: str):
        """
        Loads the model from a specified path. Default implementation loads the state_dict of the model.
        :param path: Path to load the model from
        :return:
        """
        path = os.path.join(path, 'model.pth')
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        logging.info('Model Loaded')

    @staticmethod
    @abstractmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):
        """
        Builds the model from a configuration dictionary, dataset, and feature holder.

        """

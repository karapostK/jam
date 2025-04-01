import logging
import os
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import Dataset

from data.feature import FeatureHolder


class BaseQueryMatchingModel(ABC, nn.Module):
    """
    Base class for Query Matching models.
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items

    @abstractmethod
    def forward(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the affinity scores between (u,q) and i.
        NB. it can consider also negative sampling!

        :param q_idxs:  Query indexes. Shape is (batch_size,)
        :param q_text:  Query text embedded. Shape is (batch_size, lang_dim). Where lang_dim is the language model dimension
        :param u_idxs:  User indexes. Shape is (batch_size,)
        :param i_idxs: Item indexes. Shape is (batch_size,) or (batch_size, n_neg) or (batch_size, n_items)
        :return:
            preds: Predicted scores. Shape is (batch_size,) or (batch_size, neg) or (batch_size, n_items)
        """

    def predict_all(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the affinity scores between (u,q) and all items.

        :param q_idxs:  Query indexes. Shape is (batch_size,)
        :param q_text:  Query text embedded. Shape is (batch_size, lang_dim). Where lang_dim is the language model dimension
        :param u_idxs:  User indexes. Shape is (batch_size,)
        :return:
            preds: Predicted scores. Shape is (batch_size, n_items)
        """

        # All item indexes
        i_idxs = torch.arange(self.n_items).to(q_idxs.device)
        i_idxs = i_idxs.unsqueeze(0)  # (1, n_items) -> Allowing broadcasting over batch_size

        return self.forward(q_idxs, q_text, u_idxs, i_idxs)

    def compute_loss(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor) -> dict:
        """
        Computes the loss given the predictions on positive and negative items. Default is BPR Loss

        NB. dict should have ALWAYS a single entry with key 'loss'
        :param pos_preds: Positive predictions. Shape is (batch_size,)
        :param neg_preds: Negative predictions. Shape is (batch_size, n_neg) where n_neg is the number of negative samples
        :return:
            losses: Dictionary with the loss values. IT MUST HAVE AN ENTITY 'loss'
        """

        pos_preds = pos_preds.unsqueeze(-1)  # (batch_size, 1)

        diff = pos_preds - neg_preds  # (batch_size, n_neg)

        loss_val = nn.BCEWithLogitsLoss()(
            diff,
            torch.ones_like(diff)
        )

        return {
            'loss': loss_val
        }

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

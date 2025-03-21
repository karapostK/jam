import logging

import torch
from datasets import Dataset
from torch import nn

from algorithms.base import BaseQueryMatchingModel
from data.feature import FeatureHolder
from utilities.train_utils import general_weight_init


class AverageQueryMatching(BaseQueryMatchingModel):
    """
    Simple Model for Query Matching.

    It computes the dot product between the (u + q) and i embeddings.

    # Modelling u
    Pre-trained user embeddings in FeatureHolder that are projected to dimension d.
    Embeddings are frozen

    # Modelling q
    The embedded query from a pre-trained language model is projected to dimension d.

    # Modelling i
    Item representations are retrieved from FeatureHolder, projected to dimension d, and aggregated with avg.

    """

    def __init__(self, n_users: int, n_items: int, d: int, lang_dim: int, user_features: dict, item_features: dict):
        super().__init__(n_users, n_items)

        self.d = d
        self.lang_dim = lang_dim

        # User Encoder #
        pre_train_u_embed = torch.FloatTensor(user_features['cf'])
        self.user_encoder = nn.Sequential(
            nn.Embedding.from_pretrained(pre_train_u_embed, freeze=True),
            nn.Linear(pre_train_u_embed.shape[1], d)
        )

        # Item Encoders #
        self.item_encoders = nn.ModuleDict()
        for i_feat_name, i_feat_embeds in item_features.items():
            pre_train_i_embeds = torch.FloatTensor(i_feat_embeds)
            self.item_encoders[i_feat_name] = nn.Sequential(
                nn.Embedding.from_pretrained(pre_train_i_embeds, freeze=True),
                nn.Linear(pre_train_i_embeds.shape[1], d)
            )

        # Query Encoder #
        self.query_encoder = torch.nn.Linear(self.lang_dim, d)

        # Init the model #
        self.user_encoder[1].apply(general_weight_init)
        self.query_encoder.apply(general_weight_init)
        for item_encoder in self.item_encoders.values():
            item_encoder[1].apply(general_weight_init)

        logging.info("Built AverageQueryMatching \n"
                     f"n_users: {self.n_users} \n"
                     f"n_items: {self.n_items} \n"
                     f"d: {self.d} \n"
                     f"lang_dim: {self.lang_dim} \n"
                     f"user_features: {user_features.keys()} \n"
                     f"item_features: {item_features.keys()} \n")

        logging.info(f"Parameters count: {sum(p.numel() for p in self.parameters())}")
        logging.info(f"Trainable Parameters count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                i_idxs: torch.Tensor) -> torch.Tensor:

        # Encode the queries
        q_embed = self.query_encoder(q_text)  # (batch_size, d)

        # Encode the users
        u_embed = self.user_encoder(u_idxs)  # (batch_size, d)

        # Encode the items
        i_mods_embeds = torch.stack([i_mod_encoder(i_idxs) for i_mod_encoder in self.item_encoders.values()])
        i_mods_embeds = i_mods_embeds.to(q_embed.device)  # (n_mods, batch_size, d) or # (n_mods, batch_size, n_neg, d)

        i_embed = i_mods_embeds.mean(dim=0)  # (batch_size, d) or (batch_size, n_neg, d)

        # User translation
        u_trans = q_embed + u_embed  # (batch_size, d)

        # Compute similarity
        if len(i_embed.shape) == 3:
            # In case we have n_neg or n_items
            u_trans = u_trans.unsqueeze(1)

        preds = torch.sum(u_trans * i_embed, dim=-1)  # (batch_size) or (batch_size, n_neg)

        return preds

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return AverageQueryMatching(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            d=conf['d'],
            lang_dim=conf['language_model']['hidden_size'],
            user_features=feature_holder.user_features,
            item_features=feature_holder.item_features,
        )

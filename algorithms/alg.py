import logging

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch import nn

from algorithms.base import BaseQueryMatchingModel
from data.feature import FeatureHolder
from utilities.train_utils import general_weight_init


class BaselineQueryMatching(BaseQueryMatchingModel):
    """
    Simple Model for Query Matching.

    It computes the dot product between the (u + q) and i embeddings.

    # Modelling u
    Pre-trained user embeddings in FeatureHolder that are projected to dimension d.
    Embeddings are frozen

    # Modelling q
    Query text is retrieved from FeatureHolder and encoded with e.g. SentenceT5
    The output is projected to dimension d.

    # Modelling i
    Item representations are retrieved from FeatureHolder, projected to dimension d, and aggregated with avg.

    """

    def __init__(self, n_users: int, n_items: int, d: int, user_features: dict, item_features: dict):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.d = d

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
        self.sentence_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
        self.query_encoder = torch.nn.Linear(self.sentence_model.get_sentence_embedding_dimension(), d)

        # Init the model #
        self.user_encoder[1].apply(general_weight_init)
        self.query_encoder.apply(general_weight_init)
        for item_encoder in self.item_encoders.values():
            item_encoder[1].apply(general_weight_init)

        logging.info("Built BaselineQueryMatching")
        # todo add better logging for 1)parameters count 2) # of optimizable parameters

    def forward(self, q_idxs: torch.Tensor, q_text: tuple, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:

        # Encode the queries
        q_sentence = self.sentence_model.encode(sentences=q_text,
                                                convert_to_tensor=True,
                                                batch_size=q_idxs.shape[0],
                                                show_progress_bar=False
                                                )
        q_embed = self.query_encoder(q_sentence)  # (batch_size, d)

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

    def predict_all(self, q_idxs: torch.Tensor, q_text: tuple, u_idxs: torch.Tensor) -> torch.Tensor:

        # All item indexes
        i_idxs = torch.arange(self.n_items).to(q_idxs.device)
        i_idxs = i_idxs.unsqueeze(0)  # (1, n_items) -> Allowing broadcasting over batch_size

        return self.forward(q_idxs, q_text, u_idxs, i_idxs)

    def compute_loss(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor) -> dict:

        """
        Computing BPR loss
        :param pos_preds: (batch_size,)
        :param neg_preds: (batch_size, n_neg)
        :return:
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

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return BaselineQueryMatching(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            d=conf['d'],
            user_features=feature_holder.user_features,
            item_features=feature_holder.item_features,
        )

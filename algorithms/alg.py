import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch import nn

from algorithms.base import BaseQueryMatchingModel
from data.feature import FeatureHolder


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

    def __init__(self, n_users: int, n_items: int, d: int, user_features: dict, item_features: dict,
                 queries: pd.DataFrame):
        self.n_users = n_users
        self.n_items = n_items
        self.d = d

        user_embed = user_features['cf']  # Pre-trained user embeddings
        self.user_embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(user_embed), freeze=True)
        self.user_encoder = torch.nn.Linear(user_embed.shape[1], d)

        self.queries = queries
        self.sentence_model = SentenceTransformer('sentence-transformers/sentence-t5-l')
        self.query_encoder = torch.nn.Linear(768, d)  # SentenceT5 output

        # for each item features, define embedding
        self.item_embeds = {}
        self.item_encoders = {}
        for item_feat in item_features:
            item_embed = item_features[item_feat]
            self.item_embeds[item_feat] = torch.nn.Embedding.from_pretrained(torch.FloatTensor(item_embed), freeze=True)
            self.item_encoders[item_feat] = torch.nn.Linear(item_embed.shape[1], d)

        # Init the model
        # TODO. prettify here
        self.apply(self.user_encoder)
        self.apply(self.query_encoder)
        for item_encoder in self.item_encoders.values():
            self.apply(item_encoder)

        super().__init__()

    def forward(self, q_idxs: torch.Tensor, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:

        # Encode the queries
        q_text = self.queries['text'][q_idxs]
        q_sentence = self.sentence_model.encode(sentences=q_text, convert_to_tensor=True)
        q_embed = self.query_encoder(q_sentence)

        # Encode the users
        u_embed = self.user_embed(u_idxs)
        u_embed = self.user_encoder(u_embed)

        # Encode the items
        i_embed = torch.cat([self.item_encoders[item](self.item_embeds[item](i_idxs)) for item in self.item_encoders],
                            dim=1)
        i_embed = i_embed.mean(dim=1)

        # Compute the dot product
        preds = torch.sum(q_embed * u_embed * i_embed, dim=1)
        return preds

    def predict_all(self, q_idxs: torch.Tensor, u_idxs: torch.Tensor) -> torch.Tensor:

        i_idxs = torch.arange(self.n_items).unsqueeze(0).repeat(q_idxs.shape[0], 1)
        return self.forward(q_idxs, u_idxs, i_idxs)

    def compute_loss(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor) -> dict:

        # BPR loss
        loss_func = nn.BCEWithLogitsLoss()
        return {'loss': loss_func(pos_preds.unsqueeze(1) - neg_preds, torch.ones_like(neg_preds))}

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return BaselineQueryMatching(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            d=conf['d'],
            user_features=feature_holder.user_features,
            item_features=feature_holder.item_features,
            queries=feature_holder.queries
        )

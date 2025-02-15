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

        user_embed = user_features['cf']  # Pre-trained user embeddings
        self.user_embed = torch.nn.Embedding.from_pretrained(torch.FloatTensor(user_embed), freeze=True)
        self.user_encoder = torch.nn.Linear(user_embed.shape[1], d)

        self.sentence_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
        self.query_encoder = torch.nn.Linear(768, d)  # SentenceT5 output

        # for each item features, define embedding
        self.item_embeds = nn.ModuleDict({
            item_feat: nn.Embedding.from_pretrained(
                torch.FloatTensor(item_features[item_feat]), freeze=True
            ) for item_feat in item_features
        })

        self.item_encoders = nn.ModuleDict({
            item_feat: nn.Linear(item_features[item_feat].shape[1], d)
            for item_feat in item_features
        })

        # Init the model
        # TODO. prettify here
        self.user_encoder.apply(general_weight_init)
        self.query_encoder.apply(general_weight_init)
        for item_encoder in self.item_encoders.values():
            item_encoder.apply(general_weight_init)

    def forward(self, q_idxs: torch.Tensor, q_text: tuple, u_idxs: torch.Tensor, i_idxs: torch.Tensor) -> torch.Tensor:

        #TODO: Need to check all over this again.
        # Encode the queries
        q_sentence = self.sentence_model.encode(sentences=q_text, convert_to_tensor=True,
                                                batch_size=q_idxs.shape[0],
                                                show_progress_bar=False)
        q_embed = self.query_encoder(q_sentence)  # (batch_size, d)

        # Encode the users
        u_embed = self.user_embed(u_idxs)
        u_embed = self.user_encoder(u_embed)  # (batch_size, d)

        # Encode the items

        i_reprs = []
        for item_feat_name in self.item_embeds:
            i_embed = self.item_embeds[item_feat_name](i_idxs)
            i_embed = self.item_encoders[item_feat_name](i_embed)
            i_reprs.append(i_embed)

        i_embeds = torch.stack(i_reprs).to(q_embed.device)  # [repr,..]
        i_embed = i_embeds.mean(dim=0)  # (batch_size,d) or (batch_size, n_neg, d)

        # Compute the dot product
        translation = q_embed + u_embed
        if len(i_embed.shape) == 3:
            translation = translation.unsqueeze(1)
        preds = torch.sum(translation * i_embed, dim=-1)
        return preds

    def predict_all(self, q_idxs: torch.Tensor, q_text: tuple, u_idxs: torch.Tensor) -> torch.Tensor:

        i_idxs = torch.arange(self.n_items).unsqueeze(0).to(q_idxs.device)
        return self.forward(q_idxs, q_text, u_idxs, i_idxs)

    def compute_loss(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor) -> dict:

        # pos_preds Shape is (batch_size,)
        # neg_preds Shape is (batch_size, n_neg)

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
        )

import logging
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import Dataset

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

    def predict_weight_and_score(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                                 i_idxs: torch.Tensor):
        """
        Returns the score of the model for a batch of data in the format weight * score.
        NB. i_idxs should be a 2D tensor.
        returns
            scores: (batch_size, n_mods)
            weights: (batch_size, n_mods)
        """

        assert i_idxs.ndim == 2, f"Expected 2D tensor, got {i_idxs.ndim}D"

        # Encode the queries
        q_embed = self.query_encoder(q_text)  # (batch_size, d)

        # Encode the users
        u_embed = self.user_encoder(u_idxs)  # (batch_size, d)

        # User translation
        u_trans = q_embed + u_embed  # (batch_size, d)

        # Encode the items
        i_mods_embeds = torch.stack([i_mod_encoder(i_idxs) for i_mod_encoder in self.item_encoders.values()])
        i_mods_embeds = i_mods_embeds.to(q_embed.device)  # (n_mods, batch_size, d)

        # Computing per-modality scores
        scores = torch.sum(u_trans * i_mods_embeds, dim=-1)  # (n_mods, batch_size)
        weights = torch.full(scores.shape, 1 / len(self.item_encoders), device=scores.device)  # Same as above

        scores = scores.T
        weights = weights.T

        return scores, weights

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


class CrossAttentionQueryMatching(BaseQueryMatchingModel):
    """
    The different item modalities are aggregated together with cross attention using the query text as 'query'

    # Modelling u and q
    Same as AverageQueryMatching

    # Modelling i
    Each modality (and query) is projected to another latent space of dimension d. Here, we perform cross attention.

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
        self.query_encoder = nn.Linear(self.lang_dim, d)

        # Cross Attention #
        self.w_q = nn.Linear(self.lang_dim, d)

        # Separate key encoders for each modality
        self.w_k = nn.ModuleDict()
        for i_feat_name, i_feat_encoder in self.item_encoders.items():
            self.w_k[i_feat_name] = nn.Sequential(
                i_feat_encoder[0],
                nn.Linear(i_feat_encoder[1].in_features, d)
            )

        # Init the model #
        self.user_encoder[1].apply(general_weight_init)
        self.query_encoder.apply(general_weight_init)
        for item_encoder in self.item_encoders.values():
            item_encoder[1].apply(general_weight_init)
        self.w_q.apply(general_weight_init)
        for w_k in self.w_k.values():
            w_k[1].apply(general_weight_init)

        logging.info("Built CrossAttentionQueryMatching \n"
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

        # User translation
        u_trans = q_embed + u_embed  # (batch_size, d)

        # Cross Attention
        q = self.w_q(q_text)  # (batch_size, d)
        k = torch.stack([w_k(i_idxs) for w_k in self.w_k.values()])
        k = k.to(q.device)  # (n_mods, batch_size, d) or # (n_mods, batch_size, n_neg, d)
        v = i_mods_embeds

        # Swapping dimensions for the cross attention according to docs in functional.scale_dot_product_attention
        # b = batch_size
        # n = n_neg
        # m = n_mods
        if i_idxs.dim() == 2:
            q = rearrange(q, 'b d -> b 1 1 1 d')
            k = rearrange(k, 'm b n d -> b n 1 m d')
            v = rearrange(v, 'm b n d -> b n 1 m d')
        else:
            q = rearrange(q, 'b d -> b 1 1 d')
            k = rearrange(k, 'm b d -> b 1 m d')
            v = rearrange(v, 'm b d -> b 1 m d')

        # (batch_size, n , 1, 1, d) or (batch_size, 1, 1, d)
        i_embed = F.scaled_dot_product_attention(q, k, v)
        i_embed = i_embed.squeeze(-2).squeeze(-2)  # (batch_size, n , d) or (batch_size, d)

        # Compute similarity
        if len(i_embed.shape) == 3:
            # In case we have n_neg or n_items
            u_trans = u_trans.unsqueeze(1)

        preds = torch.sum(u_trans * i_embed, dim=-1)  # (batch_size) or (batch_size, n_neg)

        return preds

    def predict_weight_and_score(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                                 i_idxs: torch.Tensor):
        """
        Returns the score of the model for a batch of data in the format weight * score.
        NB. i_idxs should be a 2D tensor.
        returns
            scores: (batch_size, n_mods)
            weights: (batch_size, n_mods)
        """

        assert i_idxs.ndim == 2, f"Expected 2D tensor, got {i_idxs.ndim}D"

        # Encode the queries
        q_embed = self.query_encoder(q_text)  # (batch_size, d)

        # Encode the users
        u_embed = self.user_encoder(u_idxs)  # (batch_size, d)

        # User translation
        u_trans = q_embed + u_embed  # (batch_size, d)

        # Encode the items
        i_mods_embeds = torch.stack([i_mod_encoder(i_idxs) for i_mod_encoder in self.item_encoders.values()])
        i_mods_embeds = i_mods_embeds.to(q_embed.device)  # (n_mods, batch_size, d)

        # Cross Attention
        q = self.w_q(q_text)  # (batch_size, d)
        k = torch.stack([w_k(i_idxs) for w_k in self.w_k.values()])
        k = k.to(q.device)  # (n_mods, batch_size, d)

        # Attention weight (looking at the implementation
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        scale_factor = 1 / math.sqrt(self.d)
        weights = torch.sum(q * k, dim=-1) * scale_factor  # (n_mods, batch_size)
        weights = torch.softmax(weights, dim=0)  # (n_mods, batch_size)

        # Computing per-modality scores
        scores = torch.sum(u_trans * i_mods_embeds, dim=-1)  # (n_mods, batch_size)

        scores = scores.T
        weights = weights.T

        return scores, weights

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return CrossAttentionQueryMatching(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            d=conf['d'],
            lang_dim=conf['language_model']['hidden_size'],
            user_features=feature_holder.user_features,
            item_features=feature_holder.item_features,
        )


class SparseMoEQueryMatching(BaseQueryMatchingModel):
    """
    As the CrossAttentionQueryMatching, however we enforce the sparse activation as in https://arxiv.org/pdf/1701.06538

    # Modelling u and q
    Same as AverageQueryMatching

    # Modelling i
    Each modality (and query) is projected to another latent space of dimension d. Here, we perform sparse cross attention
    """

    def __init__(self, n_users: int, n_items: int, d: int, lang_dim: int, user_features: dict, item_features: dict,
                 top_k: int):
        """
        :param top_k: Number of maximally active experts for an input.
        """
        super().__init__(n_users, n_items)

        assert 0 < top_k <= len(item_features), "top_k should be in the range of the number of item features"

        self.d = d
        self.lang_dim = lang_dim
        self.top_k = top_k

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
        self.query_encoder = nn.Linear(self.lang_dim, d)

        # Sparse MoE parameters #

        # Gating Network
        self.h_q = nn.Linear(self.lang_dim, d)
        self.h_k = nn.ModuleDict()
        for i_feat_name, i_feat_encoder in self.item_encoders.items():
            self.h_k[i_feat_name] = nn.Sequential(
                i_feat_encoder[0],
                nn.Linear(i_feat_encoder[1].in_features, d)
            )
        # Noise injection
        self.noise_q = nn.Linear(self.lang_dim, d)
        self.noise_k = nn.ModuleDict()
        for i_feat_name, i_feat_encoder in self.item_encoders.items():
            self.noise_k[i_feat_name] = nn.Sequential(
                i_feat_encoder[0],
                nn.Linear(i_feat_encoder[1].in_features, d)
            )

        # Init the model #
        self.user_encoder[1].apply(general_weight_init)
        self.query_encoder.apply(general_weight_init)
        for item_encoder in self.item_encoders.values():
            item_encoder[1].apply(general_weight_init)
        self.h_q.apply(general_weight_init)
        for h_k in self.h_k.values():
            h_k[1].apply(general_weight_init)
        self.noise_q.apply(general_weight_init)
        for noise_k in self.noise_k.values():
            noise_k[1].apply(general_weight_init)

        logging.info("Built SparseMoEQueryMatching \n"
                     f"n_users: {self.n_users} \n"
                     f"n_items: {self.n_items} \n"
                     f"d: {self.d} \n"
                     f"lang_dim: {self.lang_dim} \n"
                     f"user_features: {user_features.keys()} \n"
                     f"item_features: {item_features.keys()} \n"
                     f"top_k: {self.top_k} \n")

        logging.info(f"Parameters count: {sum(p.numel() for p in self.parameters())}")
        logging.info(f"Trainable Parameters count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                i_idxs: torch.Tensor) -> torch.Tensor:

        # Encode the queries
        q_embed = self.query_encoder(q_text)  # (batch_size, d)

        # Encode the users
        u_embed = self.user_encoder(u_idxs)  # (batch_size, d)

        # Encode the items
        # Here a modality is considered a different expert
        i_mods_embeds = torch.stack([i_mod_encoder(i_idxs) for i_mod_encoder in self.item_encoders.values()])
        i_mods_embeds = i_mods_embeds.to(q_embed.device)  # (n_mods, batch_size, d) or # (n_mods, batch_size, n_neg, d)

        # User translation
        u_trans = q_embed + u_embed  # (batch_size, d)

        # Sparse MoE with Noisy Top-K Gating #

        # Computing left-side of H(x) in Eq. 4 https://arxiv.org/pdf/1701.06538
        # x is q.T @ k for each modality
        q_gate = self.h_q(q_text)  # (batch_size, d)
        k_gate = torch.stack([h_k(i_idxs) for h_k in self.h_k.values()])
        k_gate = k_gate.to(q_gate.device)  # (n_mods, batch_size, d) or # (n_mods, batch_size, n_neg, d)

        if i_idxs.dim() == 2:
            q_gate = q_gate.unsqueeze(1)

        gate_activations = torch.sum(q_gate * k_gate, dim=-1)  # (n_mods, batch_size) or (n_mods, batch_size, n_neg)

        # Computing right-side of H(x) in Eq. 4 https://arxiv.org/pdf/1701.06538
        q_noise = self.noise_q(q_text)  # (batch_size, d)
        k_noise = torch.stack([noise_k(i_idxs) for noise_k in self.noise_k.values()])
        k_noise = k_noise.to(q_noise.device)  # (n_mods, batch_size, d) or # (n_mods, batch_size, n_neg, d)

        if i_idxs.dim() == 2:
            q_noise = q_noise.unsqueeze(1)

        noise_activations = torch.sum(q_noise * k_noise, dim=-1)  # (n_mods, batch_size) or (n_mods, batch_size, n_neg)
        noise_stddev = F.softplus(noise_activations)  # (n_mods, batch_size) or (n_mods, batch_size, n_neg)

        # Putting H(x) together

        # (n_mods, batch_size) or (n_mods, batch_size, n_neg)
        h_qk = gate_activations + noise_stddev * torch.randn_like(noise_stddev)

        # Applying the top_k activation
        _, topk_idxs = torch.topk(h_qk, self.top_k, dim=0)

        # Creating mask
        topk_mask = torch.zeros_like(h_qk, dtype=torch.bool)
        topk_mask.scatter_(0, topk_idxs, True)

        # Apply Mask
        masked_h_qk = h_qk.masked_fill(~topk_mask, float('-inf'))

        # G(x) = Softmax(KeepTopK(H(x), k))
        gates = F.softmax(masked_h_qk, dim=0)  # (n_mods, batch_size) or (n_mods, batch_size, n_neg)

        gated = torch.sum(gates.unsqueeze(-1) * i_mods_embeds, dim=0)  # (batch_size, d) or (batch_size, n_neg, d)

        # Compute similarity
        if i_idxs.dim() == 2:
            # In case we have n_neg or n_items
            u_trans = u_trans.unsqueeze(1)

        preds = torch.sum(u_trans * gated, dim=-1)  # (batch_size) or (batch_size, n_neg)

        return preds

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return SparseMoEQueryMatching(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            d=conf['d'],
            lang_dim=conf['language_model']['hidden_size'],
            user_features=feature_holder.user_features,
            item_features=feature_holder.item_features,
            top_k=conf['top_k']
        )

    def predict_weight_and_score(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                                 i_idxs: torch.Tensor):

        assert i_idxs.ndim == 2, f"Expected 2D tensor, got {i_idxs.ndim}D"

        # Encode the queries
        q_embed = self.query_encoder(q_text)  # (batch_size, d)

        # Encode the users
        u_embed = self.user_encoder(u_idxs)  # (batch_size, d)

        # Encode the items
        # Here a modality is considered a different expert
        i_mods_embeds = torch.stack([i_mod_encoder(i_idxs) for i_mod_encoder in self.item_encoders.values()])
        i_mods_embeds = i_mods_embeds.to(q_embed.device)  # (n_mods, batch_size, d)

        # User translation
        u_trans = q_embed + u_embed  # (batch_size, d)

        # Sparse MoE with Noisy Top-K Gating #

        # Computing left-side of H(x) in Eq. 4 https://arxiv.org/pdf/1701.06538
        # x is q.T @ k for each modality

        q_gate = self.h_q(q_text)  # (batch_size, d)
        k_gate = torch.stack([h_k(i_idxs) for h_k in self.h_k.values()])
        k_gate = k_gate.to(q_gate.device)  # (n_mods, batch_size, d)

        gate_activations = torch.sum(q_gate * k_gate, dim=-1)  # (n_mods, batch_size)

        # Computing right-side of H(x) in Eq. 4 https://arxiv.org/pdf/1701.06538
        q_noise = self.noise_q(q_text)  # (batch_size, d)
        k_noise = torch.stack([noise_k(i_idxs) for noise_k in self.noise_k.values()])
        k_noise = k_noise.to(q_noise.device)  # (n_mods, batch_size, d)

        noise_activations = torch.sum(q_noise * k_noise, dim=-1)  # (n_mods, batch_size)
        noise_stddev = F.softplus(noise_activations)  # (n_mods, batch_size)

        # Putting H(x) together

        h_qk = gate_activations + noise_stddev * torch.randn_like(noise_stddev)  # (n_mods, batch_size)

        # Applying the top_k activation
        _, topk_idxs = torch.topk(h_qk, self.top_k, dim=0)

        # Creating mask
        topk_mask = torch.zeros_like(h_qk, dtype=torch.bool)
        topk_mask.scatter_(0, topk_idxs, True)

        # Apply Mask
        masked_h_qk = h_qk.masked_fill(~topk_mask, float('-inf'))

        # G(x) = Softmax(KeepTopK(H(x), k))
        weights = F.softmax(masked_h_qk, dim=0)  # (n_mods, batch_size)

        # Computing per-modality scores
        scores = torch.sum(u_trans * i_mods_embeds, dim=-1)  # (n_mods, batch_size)

        scores = scores.T
        weights = weights.T

        return scores, weights


class TalkingToYourRecSys(BaseQueryMatchingModel):
    """
    Model to represents the adaptation of Talking to Your Recs from paper https://ceur-ws.org/Vol-3787/paper6.pdf by Oramas et al.

    Main adaptations are:
    - Query is considered as another modality of the items.
    - Contrastive Learning considers also the 'negative' items for the query.

    N.B. User is not used.
    """

    def __init__(self, n_users: int, n_items: int, d: int, lang_dim: int, item_features: dict,
                 dropout_p: float = .3, temperature: float = .2):

        super().__init__(n_users, n_items)

        self.d = d
        self.lang_dim = lang_dim
        self.dropout_p = dropout_p
        self.temperature = temperature

        # Item Encoders #
        self.item_encoders = nn.ModuleDict()
        for i_feat_name, i_feat_embeds in item_features.items():
            pre_train_i_embeds = torch.FloatTensor(i_feat_embeds)
            self.item_encoders[i_feat_name] = nn.Sequential(
                nn.Embedding.from_pretrained(pre_train_i_embeds, freeze=True),
                nn.Linear(pre_train_i_embeds.shape[1], d),
                nn.Dropout(dropout_p)
            )

        # Query Encoder #
        self.query_encoder = nn.Linear(self.lang_dim, d)

        # Init the model #
        self.query_encoder.apply(general_weight_init)
        for item_encoder in self.item_encoders.values():
            item_encoder[1].apply(general_weight_init)

        # Ignore this
        self._logging_names = None

        logging.info("Built TalkingToYourRecSys \n"
                     f"n_users: {self.n_users} \n"
                     f"n_items: {self.n_items} \n"
                     f"d: {self.d} \n"
                     f"lang_dim: {self.lang_dim} \n"
                     f"item_features: {item_features.keys()} \n"
                     f"dropout_p: {self.dropout_p} \n"
                     f"temperature: {self.temperature} \n")

        logging.info(f"Parameters count: {sum(p.numel() for p in self.parameters())}")
        logging.info(f"Trainable Parameters count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                i_idxs: torch.Tensor) -> torch.Tensor:

        """
        :return: Item embeddings for each modality . Shape is (n_mods+1, batch_size, d) or (n_mods+1, batch_size, n_neg, d)
        """

        # Encode the queries
        q_embed = self.query_encoder(q_text)  # (batch_size, d)

        # Encode the items
        i_feat_names = []  # Keeping it track just for logging purposes
        i_mods_list = []

        for i_feat_name, i_feat_encoder in self.item_encoders.items():
            i_feat_names.append(i_feat_name)
            i_mods_list.append(i_feat_encoder(i_idxs))

        i_mods_embeds = torch.stack(i_mods_list)
        i_mods_embeds = i_mods_embeds.to(q_embed.device)  # (n_mods, batch_size, d) or # (n_mods, batch_size, n_neg, d)

        # Need to do manual broadcasting
        if i_idxs.dim() == 2:
            # Expand the query on the n_neg/n_items dimension
            q_embed = q_embed.unsqueeze(1)  # (batch_size, 1 ,d)
            q_embed = q_embed.expand(-1, i_idxs.shape[1], -1)  # (batch_size, n_neg, d)
            # Expand the item embeddings on the batch_size dimension (only useful when batch_size = 1 because of broadcasting)
            i_mods_embeds = i_mods_embeds.expand(-1, q_embed.shape[0], -1, -1)  # (n_mods, batch_size, n_neg, d)
        else:
            # Expand the item embeddings on the batch_size dimension (only useful when batch_size = 1 because of broadcasting)
            i_mods_embeds = i_mods_embeds.expand(-1, q_embed.shape[0], -1)  # (n_mods, batch_size, d)

        # Adding the query as additional modality
        q_embed = q_embed.unsqueeze(0)

        i_mods_embeds = torch.cat([q_embed, i_mods_embeds], dim=0)
        # (n_mods+1, batch_size, d) or (n_mods+1, batch_size, n_neg, d)

        self._logging_names = ['query'] + i_feat_names  # just for logging purposes

        # Normalizing for cosine similarity
        i_mods_embeds = F.normalize(i_mods_embeds, p=2, dim=-1)

        return i_mods_embeds

    def predict_all(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor) -> torch.Tensor:
        # All item indexes
        i_idxs = torch.arange(self.n_items).to(q_idxs.device)
        i_idxs = i_idxs.unsqueeze(0)  # (1, n_items) -> Allowing broadcasting over batch_size

        reprs = self.forward(q_idxs, q_text, u_idxs, i_idxs)  # (n_mods+1, batch_size, n_items, d)

        q_embed = reprs[0]  # (batch_size, n_items, d)
        i_embeds = reprs[1:]  # (n_mods, batch_size, n_items, d)

        # Averaging over n_mods
        i_embeds = i_embeds.mean(dim=0)  # (batch_size, n_items, d)

        preds = torch.sum(q_embed * i_embeds, dim=-1)  # (batch_size, n_items)

        return preds

    def compute_loss(self, pos_preds: torch.Tensor, neg_preds: torch.Tensor) -> dict:
        """
        Computes the InfoNCE loss.
        Reference is https://github.com/andrebola/contrastive-mir-learning/blob/2c317fa723ef598a46dedc4c181377fc67474edc/utils.py#L191
        :param pos_preds: (n_mods+1, batch_size, d)
        :param neg_preds: (n_mods+1, batch_size, n_neg, d)
        """

        # Forgo the n_neg dimension and putting all item predictions together
        neg_preds = rearrange(neg_preds, 'm b n d -> (b n) m d')
        pos_preds = rearrange(pos_preds, 'm b d -> b m d')

        items = torch.cat([pos_preds, neg_preds], dim=0)  # (batch_size(n_neg + 1), n_mods + 1, d)

        losses = {'loss': 0}
        # To avoid memory explosion, we compute the loss in a pair-wise fashion
        n_mods = items.shape[1]
        n_items = items.shape[0]
        for i in range(n_mods - 1):
            for j in range(i + 1, n_mods):
                items_i = items[:, i, :]  # (batch_size(n_neg + 1), d)
                items_j = items[:, j, :]  # (batch_size(n_neg + 1), d)

                z = torch.cat([items_i, items_j], dim=0)  # (2 * batch_size(n_neg + 1), d)

                s = (z @ z.T) / self.temperature

                # Filling diag with -inf
                s.fill_diagonal_(float('-inf'))
                s = torch.exp(s)

                # Numerator is the diagonal of one of the 2 cross-modality matrices
                num = s[:n_items, n_items:].diag().repeat(2)  # (2 * batch_size(n_neg + 1))
                # Denominator all the other elements
                den = s.sum(dim=1)  # (2 * batch_size(n_neg + 1))

                loss = - torch.log(num / den).mean()

                losses['loss'] += loss
                losses[f'loss_{self._logging_names[i]}_{self._logging_names[j]}'] = loss

        losses['loss'] /= (n_mods * (n_mods - 1) / 2)

        return losses

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return TalkingToYourRecSys(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            d=conf['d'],
            lang_dim=conf['language_model']['hidden_size'],
            item_features=feature_holder.item_features,
            dropout_p=conf['dropout_p'],
            temperature=conf['temperature']
        )


class TwoTowerModel(BaseQueryMatchingModel):
    """
    Encodes users and queries with a classic two-tower model.
    Similar to https://roegen-recsys2024.github.io/papers/recsys2024-workshops_paper_208.pdf by Tekle et al.

    # Modelling u
    Encoder followed by NN layers

    # Modelling i
    Each modality is concatenated and followed by NN layers

    NB. Query is not used.
    """

    def __init__(self, n_users: int, n_items: int, nn_layers: list[int], user_features: dict,
                 item_features: dict):
        """
        :param nn_layers: List of integers representing the layers of the NN. Last layer is output layer.
        """
        super().__init__(n_users, n_items)
        assert len(nn_layers) > 0, "Expected at least one layer"

        self.nn_layers = nn_layers

        # User Encoder #
        pre_train_u_embed = torch.FloatTensor(user_features['cf'])
        layers = [nn.Embedding.from_pretrained(pre_train_u_embed, freeze=True)]
        previous_d = pre_train_u_embed.shape[1]
        for i, d in enumerate(nn_layers):
            layers.append(nn.Linear(previous_d, d))
            previous_d = d
            if i != len(nn_layers) - 1:
                layers.append(nn.ReLU())
        self.user_tower = nn.Sequential(*layers)

        # Item Encoders #
        # NB. Item encoder should be called before the item tower
        self.item_encoders = nn.ModuleDict()
        init_embd_size = 0
        for i_feat_name, i_feat_embeds in item_features.items():
            pre_train_i_embeds = torch.FloatTensor(i_feat_embeds)
            init_embd_size += pre_train_i_embeds.shape[1]
            self.item_encoders[i_feat_name] = nn.Embedding.from_pretrained(pre_train_i_embeds, freeze=True)

        layers = []
        previous_d = init_embd_size
        for i, d in enumerate(nn_layers):
            layers.append(nn.Linear(previous_d, d))
            previous_d = d
            if i != len(nn_layers) - 1:
                layers.append(nn.ReLU())
        self.item_tower = nn.Sequential(*layers)

        # Init the model #
        self.user_tower.apply(general_weight_init)
        self.item_tower.apply(general_weight_init)

        logging.info("Built TwoTowerModel \n"
                     f"n_users: {self.n_users} \n"
                     f"n_items: {self.n_items} \n"
                     f"nn_layers: {self.nn_layers} \n"
                     f"user_features: {user_features.keys()} \n"
                     f"item_features: {item_features.keys()} \n")

        logging.info(f"Parameters count: {sum(p.numel() for p in self.parameters())}")
        logging.info(f"Trainable Parameters count: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, q_idxs: torch.Tensor, q_text: torch.Tensor, u_idxs: torch.Tensor,
                i_idxs: torch.Tensor) -> torch.Tensor:

        # Encode the users
        u_embed = self.user_tower(u_idxs)  # (batch_size, d)

        # Encode the items
        i_mods_embeds = torch.cat([i_mod_encoder(i_idxs) for i_mod_encoder in self.item_encoders.values()], dim=-1)
        i_mods_embeds = i_mods_embeds.to(u_idxs.device)  # (batch_size, sum(d)) or # (batch_size, n_neg, sum(d))

        i_embed = self.item_tower(i_mods_embeds)  # (batch_size, d) or # (batch_size, n_neg, d)

        # Compute similarity
        if len(i_embed.shape) == 3:
            # In case we have n_neg or n_items
            u_embed = u_embed.unsqueeze(1)

        preds = torch.sum(u_embed * i_embed, dim=-1)  # (batch_size) or (batch_size, n_neg)

        return preds

    @staticmethod
    def build_from_conf(conf: dict, dataset: Dataset, feature_holder: FeatureHolder):

        return TwoTowerModel(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            nn_layers=conf['nn_layers'],
            user_features=feature_holder.user_features,
            item_features=feature_holder.item_features,
        )

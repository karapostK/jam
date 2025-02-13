import torch


def recall_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                      idx_topk: torch.Tensor = None):
    """
    Computes Recall@K
    :param logits: Logits tensor (batch_size, num_items).
    :param y_true: Ground truth binary tensor (batch_size, num_items).
    :param k: Cut-off rank (default: 10).
    :param aggr_sum: Whether to return sum over batch (default: True).
    :param idx_topk: Optional precomputed top-k indices.

    :return: Recall@k (scalar if aggr_sum=True, otherwise shape (batch_size,)).
    """

    if idx_topk is None:
        idx_topk = logits.topk(k=k, dim=-1).indices

    indexing_column = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)

    num = y_true[indexing_column, idx_topk].sum(dim=-1)

    den = y_true.sum(dim=-1)
    den = torch.clamp(den, min=1)  # Ensuring denominator is at least 1

    recall = num / den
    recall = torch.clamp(recall, 0, 1)  # For numerical stability

    return recall.sum() if aggr_sum else recall


def precision_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                         idx_topk: torch.Tensor = None):
    """
    Computes Precision@K
    :param logits: Logits tensor (batch_size, num_items).
    :param y_true: Ground truth binary tensor (batch_size, num_items).
    :param k: Cut-off rank (default: 10).
    :param aggr_sum: Whether to return sum over batch (default: True).
    :param idx_topk: Optional precomputed top-k indices.

    :return: Precision@k (scalar if aggr_sum=True, otherwise shape (batch_size,)).
    """

    if idx_topk is None:
        idx_topk = logits.topk(k=k, dim=-1).indices

    indexing_column = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)

    num = y_true[indexing_column, idx_topk].sum(dim=-1)

    precision = num / k

    return precision.sum() if aggr_sum else precision


def ndcg_at_k_batch(logits: torch.Tensor, y_true: torch.Tensor, k: int = 10, aggr_sum: bool = True,
                    idx_topk: torch.Tensor = None):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) @K
    NB. Relevance is assumed binary!

    :param logits: Logits tensor (batch_size, num_items).
    :param y_true: Ground truth binary tensor (batch_size, num_items).
    :param k: Cut-off rank (default: 10).
    :param aggr_sum: Whether to return sum over batch (default: True).
    :param idx_topk: Optional precomputed top-k indices.

    :return: NDCG@k (scalar if aggr_sum=True, otherwise shape (batch_size,)).
    """

    if idx_topk is None:
        idx_topk = logits.topk(k=k, dim=-1).indices

    indexing_column = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)

    discount_template = 1. / torch.log2(torch.arange(2, k + 2, device=logits.device).float())

    DCG = (y_true[indexing_column, idx_topk] * discount_template).sum(-1)

    IDCG = (y_true.topk(k, dim=-1).values * discount_template).sum(-1)
    IDCG = IDCG.clamp(min=1)

    NDCG = DCG / IDCG

    NDCG = NDCG.clamp(max=1.)  # Avoiding issues with the precision.

    return NDCG.sum() if aggr_sum else NDCG


def hellinger_distance(p: torch.Tensor, q: torch.Tensor):
    """
    Computes the Hellinger Distance between two probability distributions.
    It is assumed that both p and q have the same domain. The distance is symmetric.
    # https://en.wikipedia.org/wiki/Hellinger_distance
    @param p: First Probability Distribution. Shape is [*, d] where d is the discrete # of events
    @param q: Second Probability Distribution. Shape is the same as p.
    @return: Hellinger Distance. Shape is [*]
    """
    diff = torch.sqrt(p) - torch.sqrt(q)
    squared_diff = diff ** 2
    return torch.sqrt(.5 * squared_diff.sum(-1))


def kl_divergence(true_p: torch.Tensor, model_q: torch.Tensor):
    """
    Computes the Kullback-Leibler Divergence between two probability distribution. The divergence is NOT asymmetric.
    It is assumed that both p and q have the same domain.
    # https://dl.acm.org/doi/pdf/10.1145/3240323.3240372
    @param true_p: "represents the data, the observations, or a measured probability distribution" (from Wikipedia)
    @param model_q: "represents instead a theory, a model, a description or an approximation of P" (from Wikipedia)
    @return: The KL divergence of model_p from true_p.
    """
    return (true_p * (true_p.log() - model_q.log())).sum(-1)


def jensen_shannon_distance(p: torch.Tensor, q: torch.Tensor):
    """
    Computes the Jensen Shannon Distance between two probability distributions.
    It is assumed that both p and q have the same domain. The distance is symmetric.
    *NB.* The function will return nan if one of the two probability distribution is not defined on some event!
    To avoid this result, it is advised to blend each of the probability distribution with a uniform distribution over
    all the events. E.g. assuming that p is defined on 10 events: p = (1-α) * p + α * 1/d with α equal to a small value
    such as α= .01
    # https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    @param p: First Probability Distribution. Shape is [*, d] where d is the discrete # of events
    @param q: Second Probability Distribution. Shape is the same as p.
    @return: Jensen Shannon Distance. Shape is [*]
    """
    m = (.5 * (p + q))
    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)

    jsd = .5 * (kl_p_m + kl_q_m)
    return torch.sqrt(jsd)

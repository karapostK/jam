import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.base import BaseQueryMatchingModel
from eval.metrics import precision_at_k_batch, ndcg_at_k_batch, recall_at_k_batch


class Evaluator:
    """
    Helper class for the evaluation that holds the results.
    Calling eval_batch will update the internal results. After the last batch, get_results will return the metrics.

    Metrics are returned for all queries (that's it, for each data point).

    Results are either aggregated (e.g. with avg.) or returned as a list, depending on the parameter aggregate_results.
    """

    K_VALUES = [5, 10, 100]  # K value for the evaluation metrics

    def __init__(self, aggregate_results: bool = True):
        """
        :param aggregate_results: If True, the results are aggregated with mean all over queries. If not, values are
                lists of metrics
        """
        self.aggregate_results = aggregate_results

        self.metrics_dict = None
        self.n_entries = None

        self._reset_internal_dict()

    def _reset_internal_dict(self):
        self.metrics_dict = defaultdict(int) if self.aggregate_results else defaultdict(list)
        self.n_entries = 0

    def _add_entry_to_dict(self, metric_name, metric_result):
        if self.aggregate_results:
            self.metrics_dict[metric_name] += metric_result.sum().item()
        else:
            self.metrics_dict[metric_name].append(metric_result.cpu().numpy())

    def eval_batch(self, preds: torch.Tensor, pos_items: torch.Tensor):
        """
        preds: predicted scores for the items. Shape is (batch_size, n_items)
        pos_items: positive items that match the query indexes. Shape is (batch_size, n_items)
        NB. preds have items appeared in train (possibly also val) masked off with -inf
        """

        k_sorted_values = sorted(self.K_VALUES, reverse=True)
        k_max = k_sorted_values[0]
        idx_topk = preds.topk(k=k_max).indices

        self.n_entries += preds.shape[0]

        # -- Computing the Metrics --- #
        for k in k_sorted_values:
            idx_topk = idx_topk[:, :k]

            for metric_name, metric in \
                    zip(
                        ['precision@{}', 'recall@{}', 'ndcg@{}'],
                        [precision_at_k_batch, recall_at_k_batch, ndcg_at_k_batch]
                    ):
                metric_result = metric(
                    logits=preds,
                    y_true=pos_items,
                    k=k,
                    aggr_sum=False,
                    idx_topk=idx_topk
                ).detach()  # Shape is (batch_size)
                self._add_entry_to_dict(metric_name.format(k), metric_result)

    def get_results(self):

        metrics_dict = dict()
        for metric_name in self.metrics_dict:

            if self.aggregate_results:
                metrics_dict[metric_name] = self.metrics_dict[metric_name] / self.n_entries
            else:
                metrics_dict[metric_name] = np.concatenate(self.metrics_dict[metric_name])

        self._reset_internal_dict()

        return metrics_dict


@torch.no_grad()
def evaluate_algorithm(model: BaseQueryMatchingModel, eval_loader: DataLoader, device='cpu', verbose=False):
    """
    Evaluation procedure that calls Evaluator on the dataset.
    """

    evaluator = Evaluator(aggregate_results=True)
    iterator = tqdm(eval_loader) if verbose else eval_loader

    for q_idxs, q_text, u_idxs, pos_i_masks, exclude_i_masks in iterator:
        q_idxs = q_idxs.to(device)
        u_idxs = u_idxs.to(device)
        pos_i_masks = pos_i_masks.to(device)
        exclude_i_masks = exclude_i_masks.to(device)

        preds = model.predict_all(q_idxs, q_text, u_idxs)
        preds[exclude_i_masks] = -torch.inf

        evaluator.eval_batch(preds, pos_i_masks)

    metrics_values = evaluator.get_results()

    # Logging

    for metric_name, metric_value in metrics_values.items():
        logging.info(f"{metric_name} : {metric_value:.5f}")

    return metrics_values

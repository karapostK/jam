import logging
from collections import defaultdict

import torch
import wandb
from torch.utils import data
from tqdm import trange, tqdm

from algorithms.base import BaseQueryMatchingModel
from evaluation.eval import evaluate_algorithm


class Trainer:

    def __init__(self, model: BaseQueryMatchingModel,
                 train_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 conf: dict):
        """
        Train and Evaluate the model.
        :param model: Model to train
        :param train_loader: Training DataLoader
        :param val_loader: Validation DataLoader
        :param conf: Configuration dictionary
        """

        self.full_conf = conf

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = conf['device']

        self.model = model
        self.model.to(self.device)

        self.lr = conf['lr']
        self.wd = conf['wd']

        opt_map = {
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
            'adamw': torch.optim.AdamW
        }
        self.optimizer = opt_map[conf['optimizer']](self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.n_epochs = conf['n_epochs']
        self.optimizing_metric = conf['optimizing_metric']
        self.max_patience = conf['max_patience']

        self.model_path = conf['model_path']

        running_settings = conf['running_settings']
        self.use_wandb = running_settings['use_wandb']
        self.batch_verbose = running_settings['batch_verbose']

        self.best_value = None
        self.best_metrics = None
        self.best_epoch = None
        logging.info(f'Built Trainer module \n'
                     f'- n_epochs: {self.n_epochs} \n'
                     f'- device: {self.device} \n'
                     f'- optimizing_metric: {self.optimizing_metric} \n'
                     f'- model_path: {self.model_path} \n'
                     f'- optimizer: {self.optimizer.__class__.__name__} \n'
                     f'- lr: {self.lr} \n'
                     f'- wd: {self.wd} \n'
                     f'- use_wandb: {self.use_wandb} \n'
                     f'- batch_verbose: {self.batch_verbose} \n'
                     f'- max_patience: {self.max_patience} \n')

    def fit(self):
        """
        Runs the Training procedure
        """

        self.model.to(self.device)

        current_patience = self.max_patience
        log_dict = self.val()

        self.best_value = log_dict['max_optimizing_metric'] = log_dict[self.optimizing_metric]
        self.best_epoch = log_dict['best_epoch'] = -1
        self.best_metrics = log_dict

        print(f'Init - {self.optimizing_metric}={self.best_value:.4f}')

        if self.use_wandb:
            wandb.log(log_dict)

        self.model.save_model_to_path(self.model_path)

        for epoch in trange(self.n_epochs, desc='epochs'):

            self.model.train()

            if current_patience == 0:
                print('Ran out of patience. Stopping')
                break

            # --  Training -- #
            epoch_losses = defaultdict(float)

            iterator = tqdm(self.train_loader) if self.batch_verbose else self.train_loader

            for q_idxs, q_text, u_idxs, i_idxs, neg_i_idxs in iterator:
                q_idxs = q_idxs.to(self.device)
                q_text = q_text.to(self.device)
                u_idxs = u_idxs.to(self.device)
                i_idxs = i_idxs.to(self.device)
                neg_i_idxs = neg_i_idxs.to(self.device)

                pos_preds = self.model(q_idxs, q_text, u_idxs, i_idxs)
                neg_preds = self.model(q_idxs, q_text, u_idxs, neg_i_idxs)

                losses = self.model.compute_loss(pos_preds, neg_preds)

                epoch_losses.update({k: v.item() + epoch_losses[k] for k, v in losses.items()})

                losses['loss'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}

            print(f'Epoch {epoch} - Avg Train Loss {epoch_losses["loss"]:.4f} \n')  # todo possibly add more info

            # --  Validation -- #
            metrics_values = self.val()

            curr_value = metrics_values[self.optimizing_metric]
            print(f'Epoch {epoch} - Avg Val Value {curr_value:.4f} \n')

            if curr_value > self.best_value:
                self.best_value = metrics_values['max_optimizing_metric'] = curr_value
                self.best_epoch = metrics_values['best_epoch'] = epoch
                self.best_metrics = metrics_values

                print(f'Epoch {epoch} - New best model found (val value {curr_value:.4f}) \n')
                self.model.save_model_to_path(self.model_path)

                current_patience = self.max_patience  # Reset patience
            else:
                metrics_values['max_optimizing_metric'] = self.best_value
                current_patience -= 1

            # --  Logging -- #
            log_dict = {**metrics_values, **epoch_losses}

            if self.use_wandb:
                wandb.log(log_dict)

        return self.best_metrics

    def val(self):
        """
        Runs the evaluation procedure.
        :return: the dictionary of the metric values
        """
        self.model.eval()
        print('Validation started')
        metrics_values = evaluate_algorithm(self.model, self.val_loader, self.device, self.batch_verbose)
        print('Validation finished')
        return metrics_values

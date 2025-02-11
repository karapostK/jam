import typing

import wandb

from conf.conf_parser import parse_conf_file, parse_conf, save_yaml
from constants.enums import AlgorithmsEnum, DatasetsEnum
from constants.wandb_constants import PROJECT_NAME, ENTITY_NAME
from data.dataloader import get_dataloader
from data.feature import FeatureHolder
from eval.eval import evaluate_algorithm
from train.trainer import Trainer
from utilities.utils import reproducible


def run_train_val(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: typing.Union[str, dict]):
    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    if isinstance(conf, str):
        conf = parse_conf_file(conf)
    conf = parse_conf(conf, alg, dataset)

    if conf['running_settings']['use_wandb']:
        wandb.init(
            project=PROJECT_NAME,
            entity=ENTITY_NAME,
            config=conf,
            tags=[alg.name, dataset.name],
            group=f'{alg.name} - {dataset.name} - train/val',
            name=conf['time_run'],
            job_type='train/val'
        )

    reproducible(conf['running_settings']['seed'])

    train_loader = get_dataloader(conf, 'train')
    val_loader = get_dataloader(conf, 'val')

    features = FeatureHolder(conf['dataset_path'])

    alg = alg.value.build_from_conf(conf, train_loader.dataset, features)

    trainer = Trainer(alg, train_loader, val_loader, conf)

    # Validation happens within the Trainer
    metrics_values = trainer.fit()
    save_yaml(conf['model_path'], conf)

    if conf['running_settings']['use_wandb']:
        wandb.finish()

    return metrics_values, conf


def run_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf: typing.Union[str, dict]):
    print('Starting Test')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    if isinstance(conf, str):
        conf = parse_conf_file(conf)

    if conf['running_settings']['use_wandb']:
        wandb.init(
            project=PROJECT_NAME,
            entity=ENTITY_NAME,
            config=conf,
            tags=[alg.name, dataset.name],
            group=f'{alg.name} - {dataset.name} - test',
            name=conf['time_run'],
            job_type='test'
        )

    test_loader = get_dataloader(conf, 'test')
    features = FeatureHolder(conf['dataset_path'])

    alg = alg.value.build_from_conf(conf, test_loader.dataset, features)
    alg.load_model_from_path(conf['model_path'])

    metrics_values = evaluate_algorithm(alg, test_loader, conf['device'],
                                        verbose=conf['running_settings']['batch_verbose'])

    if conf['running_settings']['use_wandb']:
        wandb.log(metrics_values, step=0)
        wandb.finish()


def run_train_val_test(alg: AlgorithmsEnum, dataset: DatasetsEnum, conf_path: str):
    print('Starting Train-Val-Test')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')

    # ------ Run train and Val ------ #
    metrics_values, conf = run_train_val(alg, dataset, conf_path)
    # ------ Run test ------ #
    run_test(alg, dataset, conf)

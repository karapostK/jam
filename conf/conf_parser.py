import json
import os.path

import yaml

from constants.conf_constants import *
from constants.enums import AlgorithmsEnum, DatasetsEnum
from utilities.utils import generate_id


def parse_conf_file(conf_path: str) -> dict:
    assert os.path.isfile(conf_path), f'Configuration File {conf_path} not found!'

    with open(conf_path, 'r') as conf_file:
        try:
            print('Reading file as Yaml...')
            conf = yaml.safe_load(conf_file)
        except:
            print('Reading file as Json...')
            conf = json.load(conf_file)
    print(' --- Configuration Loaded ---')
    return conf


def save_yaml(conf_path: str, conf: dict):
    conf_path = os.path.join(conf_path, 'conf.yml')

    with open(conf_path, 'w') as conf_file:
        yaml.dump(conf, conf_file)
    print(' --- Configuration Saved ---')


def parse_conf(conf: dict, alg: AlgorithmsEnum, dataset: DatasetsEnum) -> dict:
    """
    It sets basic parameters of the configurations and provides default parameters
    """
    assert 'data_path' in conf, "Data path is missing from the configuration file"

    conf['alg'] = alg.name
    conf['time_run'] = generate_id()
    conf['dataset'] = dataset.name
    conf['data_path'] = conf['data_path']
    if 'dataset_path' not in conf:
        conf['dataset_path'] = os.path.join(conf['data_path'], conf['dataset'], 'processed')

        # Adding default parameters
    added_parameters_list = []

    if 'model_save_path' not in conf:
        conf['model_save_path'] = DEF_MODEL_SAVE_PATH
        added_parameters_list.append(f"model_save_path={conf['model_save_path']}")

    alg_dataset_folder = "{}-{}".format(alg.name, dataset.name)
    if 'sweep_id' in conf:
        intermediate_folders = f"sweeps/{conf['sweep_id']}"
    else:
        intermediate_folders = 'single_runs'

    conf['model_path'] = os.path.join(conf['model_save_path'],
                                      alg_dataset_folder,
                                      intermediate_folders,
                                      conf['time_run'])
    os.makedirs(conf['model_path'], exist_ok=True)

    if 'optimizing_metric' not in conf:
        conf['optimizing_metric'] = DEF_OPTIMIZING_METRIC
        added_parameters_list.append(f"optimizing_metric={conf['optimizing_metric']}")

    if 'running_settings' not in conf:
        conf['running_settings'] = dict()

    if 'eval_batch_size' not in conf:
        conf['eval_batch_size'] = DEF_EVAL_BATCH_SIZE
        added_parameters_list.append(f"eval_batch_size={conf['eval_batch_size']}")

    if 'seed' not in conf:
        conf['seed'] = DEF_SEED
        added_parameters_list.append(f"seed={conf['seed']}")

    if 'use_wandb' not in conf['running_settings']:
        conf['running_settings']['use_wandb'] = DEF_USE_WANDB
        added_parameters_list.append(f"use_wandb={conf['running_settings']['use_wandb']}")

    if 'eval_n_workers' not in conf['running_settings']:
        conf['running_settings']['eval_n_workers'] = DEF_EVAL_NUM_WORKERS
        added_parameters_list.append(f"eval_n_workers={conf['running_settings']['eval_n_workers']}")

    if 'batch_verbose' not in conf['running_settings']:
        conf['running_settings']['batch_verbose'] = DEF_BATCH_VERBOSE
        added_parameters_list.append(f"batch_verbose={conf['running_settings']['batch_verbose']}")

    if 'neg_train' not in conf:
        conf['neg_train'] = DEF_NEG_TRAIN
        added_parameters_list.append(f"neg_train={conf['neg_train']}")

    if 'train_batch_size' not in conf:
        conf['train_batch_size'] = DEF_TRAIN_BATCH_SIZE
        added_parameters_list.append(f"train_batch_size={conf['train_batch_size']}")

    if 'n_epochs' not in conf:
        conf['n_epochs'] = DEF_N_EPOCHS
        added_parameters_list.append(f"n_epochs={conf['n_epochs']}")
    else:
        assert conf['n_epochs'] > 0, f"Number of epochs ({conf['n_epochs']}) should be positive"

    if 'lr' not in conf:
        conf['lr'] = DEF_LEARNING_RATE
        added_parameters_list.append(f"lr={conf['lr']}")

    if 'wd' not in conf:
        conf['wd'] = DEF_WEIGHT_DECAY
        added_parameters_list.append(f"wd={conf['wd']}")

    if 'optimizer' not in conf:
        conf['optimizer'] = DEF_OPTIMIZER
        added_parameters_list.append(f"optimizer={conf['optimizer']}")
    else:
        assert conf['optimizer'] in ['adam', 'adagrad', 'adamw'], f"Optimizer ({conf['optimizer']}) not implemented"

    if 'device' not in conf:
        conf['device'] = DEF_DEVICE
        added_parameters_list.append(f"device={conf['device']}")
    else:
        assert conf['device'] in ['cpu', 'cuda'], f"Device ({conf['device']}) not available"

    if 'max_patience' not in conf:
        conf['max_patience'] = conf['n_epochs'] - 1
        added_parameters_list.append(f"max_patience={conf['max_patience']}")
    else:
        assert 0 < conf['max_patience'] < conf[
            'n_epochs'], f"Max patience {conf['max_patience']} should be between 0 and {conf['n_epochs']}"

    if 'train_n_workers' not in conf['running_settings']:
        conf['running_settings']['train_n_workers'] = DEF_TRAIN_NUM_WORKERS
        added_parameters_list.append(f"train_n_workers={conf['running_settings']['train_n_workers']}")

    if 'language_model' not in conf:
        conf['language_model'] = dict()

    if 'model_name' not in conf['language_model']:
        conf['language_model']['model_name'] = DEF_LANGUAGE_MODEL_NAME
        added_parameters_list.append(f"model_name={conf['language_model']['model_name']}")

    if 'tokenizer_name' not in conf['language_model']:
        conf['language_model']['tokenizer_name'] = DEF_LANGUAGE_MODEL_NAME
        added_parameters_list.append(f"tokenizer_name={conf['language_model']['tokenizer_name']}")

    if 'max_length' not in conf['language_model']:
        conf['language_model']['max_length'] = DEF_MAX_LEN_LANGUAGE_MODEL
        added_parameters_list.append(f"max_length={conf['language_model']['max_length']}")

    if 'batch_size' not in conf['language_model']:
        conf['language_model']['batch_size'] = DEF_BATCH_SIZE_LANGUAGE_MODEL
        added_parameters_list.append(f"batch_size={conf['language_model']['batch_size']}")

    if 'device' not in conf['language_model']:
        conf['language_model']['device'] = conf['device']
        added_parameters_list.append(f"device={conf['language_model']['device']}")

    if 'hidden_size' not in conf['language_model']:
        conf['language_model']['hidden_size'] = DEF_HIDDEN_SIZE_LANGUAGE_MODEL
        added_parameters_list.append(f"hidden_size={conf['language_model']['hidden_size']}")

    print('Added these default parameters: ', ", ".join(added_parameters_list))
    print('For more detail, see conf/conf_parser.py & constants/conf_constants.py')

    return conf

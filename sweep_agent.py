import wandb
from conf.conf_parser import parse_conf, save_yaml
from constants.enums import AlgorithmsEnum, DatasetsEnum
from data.dataloader import get_dataloader
from data.feature import FeatureHolder
from train.trainer import Trainer
from utilities.utils import reproducible


def train_val_agent():
    # Initialization and gathering hyperparameters
    run = wandb.init(job_type='train/val')
    run_id = run.id
    sweep_id = run.sweep_id

    conf = {k: v for k, v in wandb.config.items() if k[0] != '_'}

    alg = AlgorithmsEnum[conf['alg']]
    dataset = DatasetsEnum[conf['dataset']]

    conf['sweep_id'] = sweep_id
    conf = parse_conf(conf, alg, dataset)

    # Updating wandb data
    run.tags += (alg.name, dataset.name)
    wandb.config.update(conf)

    print('Starting Train-Val')
    print(f'Algorithm is {alg.name} - Dataset is {dataset.name}')
    print(f'Sweep ID is {sweep_id}')

    reproducible(conf['running_settings']['seed'])

    train_loader = get_dataloader(conf, 'train')
    val_loader = get_dataloader(conf, 'val')

    features = FeatureHolder(conf['dataset_path'])

    model = alg.value.build_from_conf(conf, train_loader.dataset, features)

    trainer = Trainer(model, train_loader, val_loader, conf)

    # Validation happens within the Trainer
    metrics_values = trainer.fit()
    save_yaml(conf['model_path'], conf)

    wandb.finish()


train_val_agent()

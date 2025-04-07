import os
import random
import socket
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from paramiko import SSHClient
from scp import SCPClient


def generate_id(prefix=None, postfix=None):
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day, dateTimeObj.hour,
                                        dateTimeObj.minute, dateTimeObj.second, dateTimeObj.microsecond)
    if prefix is not None:
        uid = prefix + "_" + uid
    if postfix is not None:
        uid = uid + "_" + postfix
    return uid


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_bests_in_sweep(sweep_id, project_base_directory: str = '.'):
    """
    It provides full interaction with wandb to gather the best models in a sweep.
    - It queries wandb for the best hyperparameter configuration in a sweep, aggregating (mean) over the seeds.
    - It downloads the best models from the remote server to the local machine, if needed
    - It adjusts the paths if necessary
    """

    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    assert sweep.state == 'FINISHED', 'Sweep has not ended yet!'

    print('There are {} runs in the sweep'.format(len(sweep.runs)))

    # Extract Sweep hyperparameters (assuming no-nested parameters)
    hps_keys = [
        k for k, v in sweep.config['parameters'].items()
        if isinstance(v, dict) and 'distribution' in v.keys() and k != 'seed'
    ]
    hps_keys = sorted(hps_keys)

    # Extracting max_optimizing_metric
    hps2max = defaultdict(list)  # Keeps list of max_optimizing_metrics
    hps2runs = defaultdict(list)  # Keeps list of runs with the same hyperparameters

    for run in sweep.runs:
        # Generating the key of the run based on the hyperparameters #
        run_key = []
        for key in hps_keys:
            run_key.append(run.config[key])
        run_key = tuple(run_key)

        hps2max[run_key].append(run.summary['max_optimizing_metric'])
        hps2runs[run_key].append(run)

    # Averaging
    hps2max = {k: np.mean(v) for k, v in hps2max.items()}

    # Finding the key that leads to the maximum value
    best_hps = max(hps2max, key=hps2max.get)

    hps_info = ' - '.join(f'{k} = {v}' for k, v in zip(hps_keys, best_hps))
    print('Best Hyperparameters: ', hps_info)
    print('Best Average Value: ', hps2max[best_hps])

    # Making sure that the runs are all on the same machine
    best_runs = hps2runs[best_hps]
    for run in best_runs:

        run_host = run.metadata['host']
        run_config = run.config

        if run_host != socket.gethostname():
            print(f'Model is on a {run_host}. Downloading it')

            # Create base directory if absent
            run_model_path = run_config['model_path']
            local_path = os.path.join(project_base_directory, run_model_path)

            Path(local_path).mkdir(parents=True, exist_ok=True)

            with SSHClient() as ssh:
                ssh.load_system_host_keys()
                ssh.connect(run_host)
                with SCPClient(ssh.get_transport()) as scp:
                    scp.get(
                        remote_path=os.path.join("jam", run_model_path),
                        local_path=os.path.dirname(local_path),
                        recursive=True
                    )

    # Returns the configs
    return [run.config for run in best_runs], best_hps


if __name__ == '__main__':
    sweep_id = 'karapost/jam/fbt096rt'
    best_configs, best_hps = fetch_bests_in_sweep(sweep_id)
    print('Best Hyperparameters: ', best_hps)
    print('Best Configs: ', best_configs)

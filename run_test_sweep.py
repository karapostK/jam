import argparse
import logging

import numpy as np

from constants.enums import AlgorithmsEnum, DatasetsEnum
from experiment_helper import run_test
from utilities.utils import fetch_bests_in_sweep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start a test experiment for a sweep')

    parser.add_argument('--sweep_id', '-s', type=str,
                        help='Complete sweep identifier. Includes entity and project names',
                        required=True)
    parser.add_argument('--eval_batch_size', '-b', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--log', type=str, default='WARNING')

    args = parser.parse_args()

    sweep_id = args.sweep_id
    eval_batch_size = args.eval_batch_size
    log = args.log

    logging.basicConfig(level=log)

    # Fetching the best configurations
    run_confs, _ = fetch_bests_in_sweep(sweep_id)

    alg = AlgorithmsEnum[run_confs[0]['alg']]
    dataset = DatasetsEnum[run_confs[0]['dataset']]

    all_experiment_results = list()

    # Running the tests
    for conf in run_confs:
        conf['eval_batch_size'] = eval_batch_size
        conf['running_settings']['batch_verbose'] = True
        metric_values = run_test(alg, dataset, conf)
        all_experiment_results.append(metric_values)

    # Calculating the mean and std of the results
    mean_std_results = dict()
    for metric in all_experiment_results[0].keys():
        mean_std_results[metric] = dict()
        mean_std_results[metric]['mean'] = np.mean([x[metric] for x in all_experiment_results])
        mean_std_results[metric]['std'] = np.std([x[metric] for x in all_experiment_results])

    # Printing the results (strings are sorted)
    mean_std_results = sorted(mean_std_results.items(), key=lambda x: (x[0].split('@')[0], float(x[0].split('@')[1])))
    print('--------------------------------')
    print('Mean and Std of the results')
    print('--------------------------------')
    for metric, values in mean_std_results:
        print(f'{metric}: {values["mean"]:.4f} ± {values["std"]:.4f}')
    print('--------------------------------')

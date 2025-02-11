import argparse
import logging

from constants.enums import AlgorithmsEnum, DatasetsEnum
from experiment_helper import run_train_val, run_test, run_train_val_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start an experiment')

    parser.add_argument('--algorithm', '-a', type=str, help='Query Matching Algorithm',
                        choices=[*AlgorithmsEnum])
    parser.add_argument('--dataset', '-d', type=str, help='Query Dataset',
                        choices=[*DatasetsEnum], required=False, default='amazon23')

    parser.add_argument('--conf_path', '-c', type=str, help='Path to the .yml containing the configuration')

    parser.add_argument('--run_type', '-t', type=str, choices=['train_val', 'test', 'train_val_test'],
                        default='train_val_test',
                        help='Type of run to carry out among "Train + Val", "Test", and "Train + Val + Test"')
    parser.add_argument('--log', type=str, default='WARNING')

    args = parser.parse_args()

    alg = AlgorithmsEnum[args.algorithm]
    dataset = DatasetsEnum[args.dataset]
    conf_path = args.conf_path
    run_type = args.run_type
    log = args.log

    logging.basicConfig(level=log)
    match run_type:
        case 'train_val':
            run_train_val(alg, dataset, conf_path)
        case 'test':
            run_test(alg, dataset, conf_path)
        case _:
            run_train_val_test(alg, dataset, conf_path)

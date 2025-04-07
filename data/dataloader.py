import logging

from torch.utils.data import DataLoader

from data.dataset import TrainQueryDataset, collate_fn_negative_sampling, EvalQueryDataset


def get_dataloader(conf: dict, split_set: str) -> DataLoader:
    """
        Returns the dataloader associated to the configuration in conf
    """

    match split_set:
        case 'train':
            train_dataset = TrainQueryDataset(data_path=conf['dataset_path'], lang_model_conf=conf['language_model'])

            dataloader = DataLoader(
                train_dataset,
                batch_size=conf['train_batch_size'],
                shuffle=True,
                num_workers=conf['running_settings']['train_n_workers'],
                collate_fn=lambda batch: collate_fn_negative_sampling(
                    batch, train_dataset.query2itemsSet,
                    train_dataset.n_items,
                    n_negs=conf['neg_train']
                )
            )

            logging.info(f"Built Train DataLoader module \n"
                         f"- batch_size: {conf['train_batch_size']} \n"
                         f"- train_n_workers: {conf['running_settings']['train_n_workers']} \n")
        case 'val':

            val_dataset = EvalQueryDataset(data_path=conf['dataset_path'], split_set='val',
                                           lang_model_conf=conf['language_model'])

            dataloader = DataLoader(
                val_dataset,
                batch_size=conf['eval_batch_size'],
                num_workers=conf['running_settings']['eval_n_workers'],
                shuffle=False
            )
            logging.info(f"Built Val DataLoader module \n"
                         f"- batch_size: {conf['eval_batch_size']} \n"
                         f"- eval_n_workers: {conf['running_settings']['eval_n_workers']} \n")

        case 'test':

            test_dataset = EvalQueryDataset(data_path=conf['dataset_path'], split_set='test',
                                            lang_model_conf=conf['language_model'])

            dataloader = DataLoader(
                test_dataset,
                batch_size=conf['eval_batch_size'],
                num_workers=conf['running_settings']['eval_n_workers'],
                shuffle=False
            )

            logging.info(f"Built Test DataLoader module \n"
                         f"- batch_size: {conf['eval_batch_size']} \n"
                         f"- eval_n_workers: {conf['running_settings']['eval_n_workers']} \n")
        case _:
            raise ValueError(f"split_set value '{split_set}' is invalid! Please choose from [train, val, test]")

    return dataloader

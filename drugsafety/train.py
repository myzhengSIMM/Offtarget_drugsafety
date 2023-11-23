

import os
import sys

sys.path.append('/home/liujin/Drugsafety/drugsafty')
from time import ctime

from argparse import ArgumentParser
from munch import DefaultMunch

from collections import defaultdict
import numpy as np
import pandas as pd

import torch
import deepchem as dc

from dataset_builder.build_dcdataset import dcdata_loader
from util.utils import checkDirectory, tqdm, create_logger
from util.metrics import get_metric_func
from util.model_utils import EarlystoppingCallback
from model.attentive_fp import AttentiveFPModel

import wandb
from rdkit import RDLogger
from deepchem.models.wandblogger import WandbLogger

import warnings
warnings.filterwarnings("ignore")

def run_training(config, logger):

    debug('Loading data...')
    debug(f'Data split type is {config.split}.')
    info(vars(config))

    dataset_file = os.path.join(config.dataset_dir, f'{config.dataset_name}.csv')
    model_dir = os.path.join(config.result_dir, f"{config.project}_{config.split}")
    tasks, all_dataset, transformers = dcdata_loader(dataset_file, featurizer='GraphConv', transformers=True,
                                                     reload=True, k_fold=5, sep=config.split)
    dataset, cv_datasets, test_dataset = all_dataset

    # training model
    fold_stats_dict = {}
    for k, (fold_train_data, fold_valid_data) in enumerate(cv_datasets):
        debug(f'Start training fold {k} / {len(cv_datasets)} ... ')
        wandb_logger = WandbLogger(name=f'random_stratified_fold_{k}', project=f'DS_{config.project}')
        fold_model_dir = os.path.join(model_dir, f'cross_validation_{k}')
        model = AttentiveFPModel(
            device = config.device,
            mode = "classification",
            num_layers = config.num_layers,
            num_timesteps = config.num_timesteps,
            graph_feat_size = config.graph_feat_size,
            dropout = config.dropout,
            n_tasks = len(tasks),
            n_classes = 2,
            batch_size = config.batch_size,
            learning_rate = config.lr,
            weight_decay = config.weight_decay,
            model_dir = fold_model_dir,
            tensorboard = True,
            log_frequency = int(fold_train_data.X.shape[0] / config.batch_size),
            wandb_logger = wandb_logger,
            logger = logger
        )
        valid_callback = EarlystoppingCallback(
            fold_valid_data,
            interval=int(fold_train_data.X.shape[0] / config.batch_size), 
            metrics=[dc.metrics.Metric(dc.metrics.balanced_accuracy_score, np.mean, name='valid_bacc')],
            save_dir=os.path.join(fold_model_dir, 'best_model'),
            save_on_minimum=False,
            patience=40 
        )
        train_callback = EarlystoppingCallback(
            fold_train_data,
            interval=int(fold_train_data.X.shape[0] / config.batch_size),
            metrics=[dc.metrics.Metric(dc.metrics.balanced_accuracy_score, np.mean, name='train_bacc')],
            save_on_minimum=False,
            patience=-1
        )

        loss = model.fit(fold_train_data, nb_epoch=config.epochs, max_checkpoints_to_keep=1, callbacks=[train_callback, valid_callback]) 
        model.restore(model_dir=os.path.join(fold_model_dir, 'best_model')) 
        debug(f'Best model step {model.get_global_step()}, i.e. epoch {model.get_global_step()*config.batch_size/fold_train_data.X.shape[0]}')
        fold_stat = evaluation(model, fold_train_data, fold_valid_data, test_dataset) 
        fold_stat = {d: fold_stat[d] for d in ['train', 'valid', 'test']}
        fold_stats_dict[k] = fold_stat 
        wandb.log(fold_stat)
        wandb_logger.finish()
        # break 

    info(fold_stats_dict)
    fold_stats_df = pd.DataFrame.from_dict({(dataset, fold): {m: fold_stats_dict[fold][dataset][m] for m in metrics}
                                        for fold in fold_stats_dict.keys()
                                        for dataset in ['train', 'valid', 'test']},
                                       orient='index')

    wandb.init(name=f'random_stratified_fold_average', project=f'DS_{config.project}')
    average_folds_scores = {}
    for key in ('valid', 'test'):
        average_folds_scores[key] = {
            "average": fold_stats_df.T[key].apply(lambda c: np.nanmean(c), axis=1).to_dict(),
            "std": fold_stats_df.T[key].apply(lambda c: np.nanstd(c), axis=1).to_dict()
        }
    info(average_folds_scores)
    wandb.log(average_folds_scores)
    wandb.finish()


def evaluation(model, fold_train_data, fold_valid_data, test_dataset):
    all_prediction = [model.predict(d)[:, :, 1] for d in (fold_train_data, fold_valid_data, test_dataset)] 
    average_stats = {}
    stats = {}
    for i, task in tqdm(enumerate(fold_train_data.tasks)):
        # if '_9606' in task: 
        _stats = {}
        k = 0 
        for key, data, prediction in zip(('train', 'valid', 'test'), (fold_train_data, fold_valid_data, test_dataset),
                                        all_prediction):
            w = data.w[:, i]
            y_true = data.y[:, i][w != 0] 
            y_pred = prediction[:, i][w != 0]
            _stats[key] = {metric: get_metric_func(metric)(y_true, y_pred) for metric in metrics}
            _stats[key]['num_pos'] = len(y_true[y_true != 0])
            _stats[key]['num_neg'] = len(y_true[y_true == 0])
            # for j in y_pred:
            #     if j > 0.5:
            #         k += 1
            # _stats[key]['hit_rate'] = k / len(y_pred) #hit_rate
        stats[task] = _stats
    
    debug(f"Writing results metrics to {os.path.join(model.model_dir, 'result_metircs.csv')}.")
    scores_df = pd.DataFrame.from_dict({(dataset, tasks): stats[tasks][dataset]
                                        for tasks in stats.keys()
                                        for dataset in stats[tasks].keys()},
                                       orient='index') 
    scores_df.to_csv(os.path.join(model.model_dir, 'result_metircs.csv'))
    for key in ('train', 'valid', 'test'):
        average_stats[key] = scores_df.T[key].apply(lambda c: np.nanmean(c), axis=1).to_dict()
    info(average_stats) 
    return average_stats


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--split', type=str, default='random_stratified',
                        choices=['random', 'scaffold', 'random_stratified'],
                        help='Method of splitting the data into train/val/test')

    parser.add_argument('--dataset_dir', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--dataset_name', type=str,
                        help='abs name of csv file of training dataset')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--project', type=str,
                        help='Directory name of where model checkpoints will be saved')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs to task')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--graph_feat_size', type=int, default=200,
                        help='Dimensionality of graph embeddings')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--num_timesteps', type=int, default=3,
                        help='Number of GRU time steps')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')

    config = parser.parse_args()
    config.device = f"cuda:{config.gpu}" if not config.no_cuda else "cpu"

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    logger = create_logger(name='train', save_dir=os.path.join(config.result_dir, f"{config.project}_{config.split}"), quiet=False)
    debug, info = logger.debug, logger.info
    metrics = ['roc', 'prc', 'recall', 'precision', 'mcc', 'bacc', 'f1']
    run_training(config, logger)

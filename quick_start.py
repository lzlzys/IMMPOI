# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
import wandb
import datetime
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset, MultiSessionsGraph, MultiSessionsGraph_train
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
import torch
from models import *
wandb.login(key="375d4be1b05ae85385c264e630934116316936e4")
def quick_start(model, dataset, config_dict, save_model=True, args=None):

    #MGCN
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    # logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=False)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    #DisenPOI
    with open(f'/home/t618141/python_code/DisenPOI/processed_data/{args.dataset}/raw/val.pkl', 'rb') as f:
        tmp = pickle.load(f)
        n_user, n_poi = pickle.load(f)
        del tmp

    dist_threshold = args.delta

    train_set = MultiSessionsGraph_train(f'/home/t618141/python_code/DisenPOI/processed_data/{args.dataset}', phrase='train', history_items_per_u=train_data.history_items_per_u, all_items=train_data.all_items, device=config['device'])
    val_set = MultiSessionsGraph(f'/home/t618141/python_code/DisenPOI/processed_data/{args.dataset}', phrase='test')
    test_set = MultiSessionsGraph(f'/home/t618141/python_code/DisenPOI/processed_data/{args.dataset}', phrase='val')
    train_graph_loader = DataLoader(train_set, args.train_batch_size, shuffle=True)
    val_graph_loader = DataLoader(val_set, args.train_batch_size, shuffle=False)
    test_graph_loader = DataLoader(test_set, args.train_batch_size, shuffle=False)
    with open(f'/home/t618141/python_code/DisenPOI/processed_data/{args.dataset}/processed/dist_graph_{dist_threshold}.pkl',
              'rb') as f:
        dist_edges = torch.LongTensor(pickle.load(f))
        dist_nei = pickle.load(f)

    dist_vec = np.load(f'/home/t618141/python_code/DisenPOI/processed_data/{args.dataset}/dist_on_graph_{dist_threshold}.npy')
    device = torch.device('cpu') if args.gpu is None else torch.device(f'cuda:{args.gpu}')


    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    # nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #
    # wandb.init(project='MMD', name=f'{nowtime}_{config["dataset"]}_{config["seed"]}', config=config)

    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization

        model = get_model(config['model'])(config, train_data, n_user, n_poi, dist_edges, dist_nei, args.embedding_size, args.gcn_num, dist_vec, device).to(config['device'])


        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, train_graph_data=train_graph_loader, val_graph_data=val_graph_loader, test_graph_data=test_graph_loader, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))
        # wandb.log({"best_valid_result": np.mean(dict2str(hyper_ret[best_test_idx][1]).item()), "best_test_result": np.mean(dict2str(hyper_ret[best_test_idx][1]).item())})
    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

    wandb.finish()
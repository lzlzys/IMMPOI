# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start


os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='PA', help='name of datasets')
    parser.add_argument('--delta', type=str, default='1', help='Disntance graph threshold.')
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--gcn_num', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0,
                     help='Denote training device.')
    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, args=args)



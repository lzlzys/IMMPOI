# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
import pickle


class MultiSessionsGraph_train(InMemoryDataset):
    def __init__(self, root='../processed_data/foursquare', phrase='train', transform=None, pre_transform=None, history_items_per_u=None, all_items=None, device=None):
        assert phrase in ['train', 'test', 'val', '0.2', '0.4', '0.6', '0.8', 'vis', 'vis_eval']
        self.phrase = phrase
        self.device = device
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.all_items = all_items
        self.history_items_per_u = history_items_per_u
        super(MultiSessionsGraph_train, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.pkl']

    @property
    def processed_file_names(self):
        return [self.phrase + '_session_graph_' + '.pt']

    def download(self):
        pass

    def _random(self):
        rd_id = random.sample(self.all_items, 1)[0]
        return rd_id

    def _sample_neg_ids(self, u_ids):
        neg_ids = []

        # random 1 item
        iid = self._random()
        while iid in self.history_items_per_u[u_ids]:
            iid = self._random()
        neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def process(self):
        with open(self.raw_dir + '/' + self.raw_file_names[0], 'rb') as f:
            data = pickle.load(f)

            n_user, n_poi = pickle.load(f)
        data_list = []

        for uid, poi, sequences, location, y in tqdm(data):
            i, x, senders, nodes, x_1 = 0, [], [], {}, []
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                x_1.append(x)
                senders.append(nodes[node])
            neg = self._sample_neg_ids(uid).to(self.device)
            edge_index = torch.LongTensor([senders[: -1], senders[1:]])
            x = torch.LongTensor(x)
            y = torch.LongTensor([y])
            uid = torch.LongTensor([uid])
            poi = torch.LongTensor([poi])
            location = torch.LongTensor(location)
            x_1 = torch.LongTensor([x_1])
            data_list.append(Data(x=x, edge_index=edge_index, num_nodes=len(nodes),
                                  y=y, uid=uid, poi=poi, location=location, neg=neg))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

class MultiSessionsGraph(InMemoryDataset):
    def __init__(self, root='../processed_data/foursquare', phrase='train', transform=None, pre_transform=None, history_items_per_u=None, all_items=None, device=None):
        assert phrase in ['train', 'test', 'val', '0.2', '0.4', '0.6', '0.8', 'vis', 'vis_eval']
        self.phrase = phrase
        self.device = device
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.all_items = all_items
        self.history_items_per_u = history_items_per_u
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '.pkl']

    @property
    def processed_file_names(self):
        return [self.phrase + '_session_graph_' + '.pt']

    def download(self):
        pass

    def _random(self):
        rd_id = random.sample(self.all_items, 1)[0]
        return rd_id

    def _sample_neg_ids(self, u_ids):
        neg_ids = []

        # random 1 item
        iid = self._random()
        while iid in self.history_items_per_u[u_ids]:
            iid = self._random()
        neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def process(self):
        with open(self.raw_dir + '/' + self.raw_file_names[0], 'rb') as f:
            data = pickle.load(f)

            n_user, n_poi = pickle.load(f)
        data_list = []

        for uid, poi, sequences, location, y in tqdm(data):
            i, x, senders, nodes = 0, [], [], {}
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            edge_index = torch.LongTensor([senders[: -1], senders[1:]])
            x = torch.LongTensor(x)
            y = torch.LongTensor([y])
            uid = torch.LongTensor([uid])
            poi = torch.LongTensor([poi])
            location = torch.LongTensor(location)

            data_list.append(Data(x=x, edge_index=edge_index, num_nodes=len(nodes),
                                  y=y, uid=uid, poi=poi, location=location))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']

        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']
        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            # print(self.dataset_path)
            # print(i)
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        print(self.item_num)
        self.user_num = int(max(self.df[self.uid_field].values)) + 1
        print(self.user_num)
    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label, 'stars', 'latitude', 'longitude']
        self.df = pd.read_csv(inter_file)
        # self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        # print(self.df)
        # print(1132)
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)

        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)
        # print('dsf')
        # print(dfs)
        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]

        return full_ds

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        # print(self.df.iloc[idx])
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)

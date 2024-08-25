# coding: utf-8
# @email: y463213402@gmail.com
r"""
MGCN
################################################
Reference:
    https://github.com/demonph10/MGCN
    ACM MM'2023: [Multi-View Graph Convolutional Network for Multimedia Recommendation]
    https://arxiv.org/abs/2308.03588
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class SeqQuery_Multi(nn.Module):
    def __init__(self, hidden_size):
        super(SeqQuery_Multi, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha = nn.Linear(self.hidden_size, 1)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def forward(self, sess_embed, query):
        weight = self.alpha(torch.sigmoid(self.W_1(query) + self.W_2(sess_embed)))
        s_g_whole = weight * sess_embed

        return s_g_whole


class SeqQuery(nn.Module):
    def __init__(self, hidden_size):
        super(SeqQuery, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha = nn.Linear(self.hidden_size, 1)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def forward(self, sess_embed, query, sections):
        v_i = torch.split(sess_embed, sections)
        q_n_repeat = tuple(query[i].view(1, -1).repeat(nodes.shape[0], 1) for i, nodes in enumerate(v_i))
        weight = self.alpha(torch.sigmoid(self.W_1(torch.cat(q_n_repeat, dim=0)) + self.W_2(sess_embed)))
        s_g_whole = weight * sess_embed
        s_g_split = torch.split(s_g_whole, sections)
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        s_h = torch.cat(s_g, dim=0)
        return s_h
class MGCN(GeneralRecommender):
    def __init__(self, config, dataset, n_user, n_poi, dist_edges, dist_nei, embed_dim, gcn_num, dist_vec, device):
        super(MGCN, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.args = config
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()

        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)

            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()


        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_embed = SeqQuery(self.embedding_dim)
        self.text_embed = SeqQuery(self.embedding_dim)


        #DisenPOI
        self.n_user, self.n_poi = n_user, n_poi
        self.embed_dim = embed_dim
        self.gcn_num = gcn_num
        self.dist_edges = to_undirected(dist_edges, num_nodes=n_poi).to(device)
        self.dist_nei = dist_nei
        self.device = device

        self.poi_embed = nn.Parameter(torch.empty(n_poi, embed_dim))
        self.user_embed = nn.Parameter(torch.empty(n_user, embed_dim))
        nn.init.xavier_normal_(self.poi_embed)
        nn.init.xavier_normal_(self.user_embed)

        self.edge_index, self.adj_weight = get_laplacian(self.dist_edges, normalization='sym', num_nodes=n_poi)
        self.adj_weight = self.adj_weight.to(device)
        self.edge_index = self.edge_index.to(device)

        dist_vec /= dist_vec.max()
        self.dist_vec = torch.cat([torch.Tensor(dist_vec), torch.Tensor(dist_vec), torch.zeros((n_poi,))]).to(device)

        self.sess_conv = GatedGraphConv(self.embed_dim, num_layers=1)
        self.geo_conv = []
        for _ in range(self.gcn_num):
            # self.geo_conv.append(GCN_layer(self.embed_dim, self.embed_dim, device))
            self.geo_conv.append(Geo_GCN(self.embed_dim, self.embed_dim, device))

        self.sess_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.geo_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.geo_proj_I = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.geo_proj_V = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.I_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.T_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.CL_builder = Contrastive_BPR()

        self.image_MLP = nn.Sequential(
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim,self.embed_dim)
            )
        self.text_MLP = nn.Sequential(
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim,self.embed_dim)
            )
        self.text_geo_MLP = nn.Sequential(
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim,self.embed_dim)
            )
        self.image_geo_MLP = nn.Sequential(
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim,self.embed_dim)
            )
        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)
        self.j = 0
        self.sess_embed = SeqQuery(self.embed_dim)
        self.geo_embed = SeqQuery(self.embed_dim)
        self.I_embed = SeqQuery_Multi(self.embed_dim)
        self.T_embed = SeqQuery_Multi(self.embed_dim)
        self.nhead = 2
        self.nlayers=2
        self.nhid = 1024
        self.Transformer = TransformerModel(n_poi, self.embed_dim, self.nhead, self.nhid, self.nlayers)
    def pre_epoch_processing(self):
        pass

    def split_mean(self, section_feat, sections):
        section_embed = torch.split(section_feat, sections)
        mean_embeds = [torch.mean(embeddings, dim=0) for embeddings in section_embed]
        return torch.stack(mean_embeds)

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)

            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, data, train=False):

        sess_idx, edges, batch_idx, uid, tar_poi = data.x.squeeze(), data.edge_index, data.batch, data.uid, data.poi
        sections = tuple(torch.bincount(batch_idx).cpu().numpy())

        # Generate geometric encoding & pooling
        geo_feat = self.item_id_embedding.weight
        dist_weight = torch.exp(-(self.dist_vec ** 2))
        for i in range(self.gcn_num):
            geo_feat = self.geo_conv[i](geo_feat, self.edge_index, dist_weight)
            geo_feat = F.leaky_relu(geo_feat)
            geo_feat = F.normalize(geo_feat, dim=-1)
        geo_enc = self.geo_embed(geo_feat[sess_idx], geo_feat[tar_poi], sections)
        geo_enc_p = self.geo_proj(geo_enc)

        geo_pool = self.geo_proj(self.split_mean(Neigh_pooling()(self.poi_embed, self.edge_index)[sess_idx], sections))#disentangle
        # Generate session encoding & pooling
        sess_hidden = F.leaky_relu(self.sess_conv(self.poi_embed[sess_idx], edges))
        sess_enc = self.sess_embed(sess_hidden, self.poi_embed[tar_poi], sections)
        sess_enc_p = self.sess_proj(sess_enc)
        sess_pool = self.sess_proj(self.split_mean(self.poi_embed[sess_idx], sections))

        # CL loss for disentanglement
        con_loss = self.CL_builder(geo_enc, geo_pool, sess_pool) + \
                   self.CL_builder(sess_enc, sess_pool, geo_pool)

        tar_embed = self.poi_embed[tar_poi]
        tar_geo_embed = geo_feat[tar_poi]


        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        # Behavior-Guided Purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))
        # User-Item View 改這裏
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)


        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        # Item-Item View Cross Attention
        I_enc = self.I_embed(text_item_embeds, image_item_embeds)
        I_enc_pool = self.geo_proj_I(Neigh_pooling()(image_item_embeds, self.edge_index)[sess_idx])
        T_enc = self.T_embed(image_item_embeds, text_item_embeds)
        T_enc_pool = self.geo_proj_V(Neigh_pooling()(text_item_embeds, self.edge_index)[sess_idx])

        text_user_embeds[uid] = self.text_geo_MLP(torch.cat((text_user_embeds[uid], sess_enc), dim=-1)).squeeze()
        print(text_user_embeds[uid].shape)
        print(sess_enc.shape)
        print(text_user_embeds[uid].shape)
        text_item_embeds = self.text_MLP(torch.cat((text_item_embeds, geo_feat), dim=-1)).squeeze()
        print(text_item_embeds.shape)
        print(geo_feat.shape)
        print(text_item_embeds.shape)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        image_user_embeds[uid] = self.image_geo_MLP(torch.cat((image_user_embeds[uid], sess_enc), dim=-1)).squeeze()
        image_item_embeds = self.image_MLP(torch.cat((image_item_embeds, geo_feat), dim=-1)).squeeze()
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Behavior-Aware Fuser
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3
        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)


        return I_enc, T_enc, I_enc_pool, T_enc_pool, all_embeddings_users, all_embeddings_items, geo_enc_p, sess_enc_p, tar_embed, tar_geo_embed, side_embeds, content_embeds, con_loss, None

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss


    def Crossentrophy(self, user_embedding, pos_embedding, neg_embedding):
        pos_scores = torch.sum(user_embedding * pos_embedding, dim=1)
        pos_probs = torch.sigmoid(pos_scores)
        # 负样本的预测
        neg_scores = torch.sum(user_embedding * neg_embedding, dim=1)
        neg_probs = torch.sigmoid(neg_scores)

        # 计算交叉熵损失
        pos_loss = F.binary_cross_entropy(pos_probs, torch.ones_like(pos_probs))
        neg_loss = F.binary_cross_entropy(neg_probs, torch.zeros_like(neg_probs))
        # 总损失
        loss = pos_loss + neg_loss
        return loss
    def top_k_acc_last_timestep(self, y_true_seq, y_pred_seq, k):
        """ next poi metrics """
        y_true = y_true_seq

        y_pred = y_pred_seq[-1]
        top_k_rec = y_pred.argsort()[-k:][::-1]

        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            return 1
        else:
            return 0


    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, graph_data):

        users, pos_items, neg_items, batch, x = graph_data.uid, graph_data.poi, graph_data.neg, graph_data.batch, graph_data.x.squeeze()
        I_enc, T_enc, I_enc_pool, T_enc_pool, ua_embeddings, ia_embeddings, geo_enc_p, sess_enc_p, tar_embed, tar_geo_embed, side_embeds, content_embeds, con_loss, y_POI = self.forward(
        self.norm_adj, graph_data, train=True)#訓練其實是沒有傳入數據的
        I_enc = I_enc[pos_items]
        T_enc = T_enc[pos_items]
        I_enc_pool = I_enc_pool[pos_items]
        T_enc_pool = T_enc_pool[pos_items]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2) + self.InfoNCE(
            I_enc, I_enc_pool, 0.2) + self.InfoNCE(T_enc, T_enc_pool, 0.2)



        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + 0.00001 * con_loss.mean(-1)

    def full_sort_predict(self, interaction, graph_data):
        user = graph_data.uid
        I_enc, T_enc, I_enc_pool, T_enc_pool, restore_user_e, restore_item_e, geo_enc_p, sess_enc_p, tar_embed, tar_geo_embed, side_embeds, content_embeds, con_loss, y_POI = self.forward(self.norm_adj, graph_data)

        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))


        return scores

from torch_geometric.utils import add_self_loops, to_undirected, degree, get_laplacian
from torch_geometric.nn import MessagePassing, GatedGraphConv


class Neigh_pooling(MessagePassing):
    def __init__(self, aggr='mean'):
        super(Neigh_pooling, self).__init__(aggr=aggr)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class Geo_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(Geo_GCN, self).__init__()
        self.W0 = nn.Linear(in_channels, out_channels, bias=False).to(device)
        self.W1 = nn.Linear(in_channels, out_channels, bias=False).to(device)
        self.W2 = nn.Linear(in_channels, out_channels, bias=False).to(device)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    def forward(self, x, edge_index, dist_weight):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        dist_adj = torch.sparse_coo_tensor(edge_index, dist_weight * norm)
        side_embed = torch.sparse.mm(dist_adj, x)

        bi_embed = torch.mul(x, side_embed)
        return self.W0(side_embed) + self.W1(bi_embed)


class GCN_layer(MessagePassing):
    def __init__(self, in_channels, out_channels, device):
        super(GCN_layer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels).to(device)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, dist_weight=None):
        edge_index, _ = add_self_loops(edge_index)
        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, embed_dim]
        return norm.view(-1, 1) * x_j


class Contrastive_BPR(nn.Module):
    def __init__(self, beta=1):
        super(Contrastive_BPR, self).__init__()
        self.Activation = nn.Softplus(beta=beta)

    def forward(self, x, pos, neg):
        loss_logit = (x * neg).sum(-1) - (x * pos).sum(-1)
        return self.Activation(loss_logit)


def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len,)

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi_1 = nn.Linear(embed_size, embed_size)
        self.decoder_poi_2 = nn.Linear(embed_size // 2, num_poi)



        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi_1.bias.data.zero_()
        self.decoder_poi_1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi_2.bias.data.zero_()
        self.decoder_poi_2.weight.data.uniform_(-initrange, initrange)
    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)

        x = x[:, -1, :]

        out_poi = self.decoder_poi_1(x)
        # out_poi = self.decoder_poi_2(out_poi)
        return out_poi


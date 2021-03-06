from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from dataloader import gen_edges
np.random.seed(123)
flags = tf.app.flags
FLAGS = flags.FLAGS


class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """

    def __init__(self, G, id2idx, labels,
                 placeholders, context_pairs=None, batch_size=100, max_degree=25,
                 n2v_retrain=False, fixed_n2v=False,
                 **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.idx2id = dict(zip(id2idx.values(), id2idx.keys()))  # idx to id
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        # self.labels = labels
        self.i = 0
        self.all_edge = gen_edges(FLAGS.time_step)
        # self.all_edge = pd.read_pickle('paper_paper.pkl')
        self.paper_venue = pd.read_pickle('paper_venue.pkl')
        self.node_classify = FLAGS.node_pred
        self.label_classes = self.paper_venue['new_venue_id'].unique()  # 43類
        self.all_edge = self.all_edge[self.all_edge['rel'] == 0][['head', 'tail']]
        # self.all_edge = self.all_edge[self.all_edge['time_step'] < 280][['new_papr_id', 'new_cited_papr_id']].reset_index(drop=True)
        self.all_edge_array = self.all_edge.values

        #        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()

        self.test_adj = self.construct_test_adj()

        #        if context_pairs is None:
        #            edges = G.edges()
        #        else:
        #            edges = context_pairs
        #        self.train_edges = self.edges = np.random.permutation(edges)
        #        self.train_edges = edges
        #        if not n2v_retrain:
        #            self.train_edges = self._remove_isolated(self.train_edges)
        #            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        #        else:
        #            if fixed_n2v:
        #                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
        #            else:
        #                self.train_edges = self.val_edges = self.edges

        # print(len([n for n in G.nodes]), 'train nodes')
        # if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        # print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')

    #        self.val_set_size = len(self.val_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n: self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0):
                continue
            else:
                new_edge_list.append((n1, n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    '''建相鄰矩陣'''

    def construct_adj(self):
        '''
        adj維度是node數*max_degree，max_degree是跟這個node相連的個數，預設是100，每一個row就是自己node的鄰居
        deg維度是node
        '''
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))
        '''
        每一個node找neighbor
        如果neighbor數大於max_degree就sample max_degree數的neighbor
        如果neighbor數小於max_degree就重複sample max_degree數的自己的neighbor
        '''
        for nodeid in self.G.nodes():
            #            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
            #                continue
            neighbors = np.array([self.id2idx[neighbor] for neighbor in self.G.neighbors(nodeid) if
                                  (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                                  for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def test(self):
        return len(self.train_edges)[:10], self.train_label[:10]

    def batch_feed_dict(self, batch_edges, save=False):
        batch1 = []
        batch2 = []
        labels = []

        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
            if not save:
                labels.append(self.train_label[self.i])
                self.i += 1

        # 確保送進validation的data不要跟train的重複
        self.batch1_data = batch1
        self.batch2_data = batch2
        self.labels_data = labels
        self.batch_edges = batch_edges
        if not save:
            labels = np.vstack(labels)
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        if not save:
            feed_dict.update({self.placeholders['labels']: labels})
            return feed_dict, labels
        else:
            return feed_dict

    def val_batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []

        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
            self.i += 1

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    # train幾個epoch就算一下accuracy
    def val_shuffle(self):
        batch_data = np.zeros((len(self.batch_edges), 2))
        batch_data[:, 0] = self.batch1_data
        batch_data[:, 1] = self.batch2_data
        batch_data_edge = set(map(tuple, batch_data))

        # todo positive的 sample來自 test的時間
        ppedge = (list(set(np.unique(self.all_edge_array[:, 0])).union(set(np.unique(self.all_edge_array[:, 1])))))
        sample_negative_edge = np.random.choice(ppedge, (4096, 2))

        # sample_negative_edge = np.random.randint(0,np.max(self.all_edge_array),(4096,2))
        idx = np.random.randint(len(self.all_edge_array), size=4096)

        sample_positive_edge = self.all_edge_array[idx, :]
        positive_edge = set(map(tuple, sample_positive_edge))
        positive_edge_without_train = positive_edge.difference(batch_data_edge)
        sample_negative_edge = set(map(tuple, sample_negative_edge))
        negative_edge = sample_negative_edge.difference(positive_edge)
        negative_edge_without_train = negative_edge.difference(batch_data_edge)

        if not self.node_classify:
            positive_edge_label = np.c_[np.array(list(positive_edge_without_train)), np.ones(len(positive_edge_without_train))]
            negative_edge_label = np.c_[np.array(list(negative_edge_without_train)), np.zeros(len(negative_edge_without_train))]
            data_edge = np.concatenate((positive_edge_label, negative_edge_label), axis=0)
            np.random.shuffle(data_edge)
            train_edges = data_edge[:, :2]
            train_label = data_edge[:, -1]
        else:
            heads, counts = np.unique(np.array(list(positive_edge_without_train))[:, 0], return_counts=True)
            head_venue = self.paper_venue[self.paper_venue['new_papr_id'].isin(heads)]['new_venue_id']
            train_edges = np.array(list(positive_edge_without_train))
            train_label = pd.get_dummies(np.repeat(head_venue, counts)).values

        batch_edges = train_edges

        batch1 = []
        batch2 = []
        labels = []
        i = 0

        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
            labels.append(train_label[i])
            i += 1

        labels = np.vstack(labels)
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx: end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.train_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num * size:min((iter_num + 1) * size, len(edge_list))]

        # return self.batch_feed_dict(val_edges), (iter_num + 1) * size >= len(self.val_edges), self.id2idx[val_edges]
        return self.batch_feed_dict(val_edges), (iter_num + 1) * size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = np.array(self.nodes)
        # node_list = np.arange(0, len(self.nodes))
        val_nodes = node_list[iter_num * size:min((iter_num + 1) * size, len(node_list))]
        # val_nodes_mapping = list(map(self.idx2id.get, val_nodes))  # map idx to id
        val_edges = [(n, n) for n in val_nodes]
        return self.batch_feed_dict(val_edges, True), (iter_num + 1) * size >= len(node_list), val_edges
        # return self.val_batch_feed_dict(val_edges), (iter_num + 1) * size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test']
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1, n2))
            else:
                train_edges.append((n1, n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        #        self.train_edges = np.random.permutation(self.train_edges)
        #        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0
        # ppedge = (list(set(np.unique(self.all_edge_array[:, 0])).union(set(np.unique(self.all_edge_array[:, 1])))))  # all paper id
        self.ppedge = gen_edges(FLAGS.time_step, pp_only=True)[['head', 'tail']].values
        train_nums = int(round(len(self.ppedge) * 0.8))  # 保留一部分當 validation
        self.ppedge = self.ppedge[:train_nums]
        ppedge = np.unique(self.ppedge.flatten())
        neg_size = FLAGS.neg_size  # default is 2
        sample_edge = np.random.choice(ppedge, (len(self.all_edge_array) * neg_size, 2))

        # self.positive_edge = set(map(tuple, self.all_edge_array))
        # self.sample_negative_edge = set(map(tuple, sample_edge))
        # self.negative_edge = self.sample_negative_edge.difference(self.positive_edge)
        self.positive_edge = set(map(tuple, self.ppedge))
        self.sample_negative_edge = set(map(tuple, sample_edge))
        self.negative_edge = self.sample_negative_edge.difference(self.positive_edge)

        if not self.node_classify:
            # assign true/ false, edge list + label, id還沒轉成idx
            # self.positive_edge_label = np.c_[self.all_edge_array, np.ones(len(self.all_edge_array))]

            self.positive_edge_label = np.c_[self.ppedge, np.ones(len(self.ppedge))]
            self.negative_edge_label = np.c_[np.array(list(self.negative_edge)), np.zeros(len(self.negative_edge))]
            self.data_edge = np.concatenate((self.positive_edge_label, self.negative_edge_label), axis=0)
            np.random.shuffle(self.data_edge)
            self.train_edges = self.data_edge[:, :2]
            self.train_label = self.data_edge[:, -1]
        else:
            # find each id class/ venue, 這裡還是舊的id, 沒有neg sample的問題
            heads, counts = np.unique(self.all_edge_array[:, 0], return_counts=True)
            head_venue = self.paper_venue[self.paper_venue['new_papr_id'].isin(heads)]['new_venue_id']
            self.train_edges = self.all_edge_array
            # label轉 one hot
            self.train_label = pd.get_dummies(np.repeat(head_venue, counts)).values
        self.i = 0

    def before_train(self):
        batch_edges = self.train_edges[:100000]
        batch1 = []
        batch2 = []
        labels = []
        i = 0

        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
            labels.append(self.train_label[i])
            i += 1

        labels = np.vstack(labels)
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """

    def __init__(self, G, id2idx,
                 placeholders, label_map, num_classes,
                 batch_size=100, max_degree=25,
                 **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                                  for neighbor in self.G.neighbors(nodeid)
                                  if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                                  for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num * size:min((iter_num + 1) * size,
                                                        len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num + 1) * size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num * size:min((iter_num + 1) * size,
                                                  len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num + 1) * size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

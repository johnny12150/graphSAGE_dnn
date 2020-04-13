from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    '''
    第一次的input shape是node數*max_degree(28930*100)
    '''
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
#        print(adj_lists.shape)
#        adj_lists = tf.transpose(adj_lists)
#        print(adj_lists.shape)
#        adj_lists = tf.random_shuffle(adj_lists)
#        print(adj_lists.shape)
#        adj_lists = tf.transpose(adj_lists)
#        print(adj_lists.shape)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        # print(adj_lists.shape)
        return adj_lists
        '''
        第一次adj_lists的維度是(?,10)，第二次的維度是(20,10)
        <tf.Tensor 'uniformneighborsampler_1_4/Slice:0' shape=(20, 10) dtype=int32>
        <tf.Tensor 'uniformneighborsampler_1_5/embedding_lookup/Identity:0' shape=(200, 100) dtype=int32>
        '''
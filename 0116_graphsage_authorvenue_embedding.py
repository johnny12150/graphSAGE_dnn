# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:17:27 2020

@author: rsps971130
"""

import pickle 
import numpy as np
import pandas as pd
import networkx as nx
from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
import tensorflow as tf
import os
import time
import random

#%%
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string("model_size", "big", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')  # 2-hop neighbors
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')  # 1-hop neighbors
flags.DEFINE_integer('dim_1', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 4096, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 50, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_boolean('node_pred', False, 'Which task to perform')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', 'embedding', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 500, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 4096, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 100, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10000, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

#%%
#all_edge = pd.read_pickle('new_av')
#all_edge = all_edge[['new_author_id','new_venue_year_id']]
#all_edge_array = np.array(all_edge)
#sample_edge = np.random.randint(0,np.max(all_edge_array),(1000000,2))
#positive_edge = set(map(tuple, all_edge_array))
#sample_negative_edge = set(map(tuple, sample_edge))
#negative_edge = sample_negative_edge.difference(positive_edge)
#
#positive_edge_label = np.c_[all_edge_array, np.ones(len(all_edge_array))]
#negative_edge_label = np.c_[np.array(list(negative_edge)), np.zeros(len(negative_edge))]
#
#data_edge = np.concatenate((positive_edge_label, negative_edge_label), axis=0)
#np.random.shuffle(data_edge)
#train_edge = data_edge[:, :2]
#train_edge_label = data_edge[:, -1]

#%%
#G = nx.DiGraph()
#G.add_edges_from(train_edge)
#id_map = dict(zip(G.nodes(), np.arange(len(G.nodes())))) 
#walks = G.edges()
#feats = None

def load_data(prefix, normalize=True, load_walks=None, time_step=None, draw_G=True, save_fig='graph'):
#    assert time_step != None, "load_data-- time_step can't None!"
    
    all_edge = pd.read_pickle('all_edge.pkl')
    # pv = pd.read_pickle('F:/volume/jicai/preprocess/edge/paper_venue.pkl')
    # 改成不考慮時間的 venue
    # pv = pv[pv['time_step'] < 284][['new_papr_id', 'new_venue_id']]
    # all_edge = all_edge[all_edge['rel'].isin([0, 1])]
    all_edge = all_edge[['head','tail']]
    # all_edge = np.vstack((all_edge.values, pv.values))

    G = nx.DiGraph()
    G.add_edges_from(all_edge.values)

    id_map = dict(zip(G.nodes(), np.arange(len(G.nodes())))) 
    walks = G.edges()
    feats = None
    
    for edge in G.edges():
        G[edge[0]][edge[1]]['train_removed'] = False
    
    return G, feats, id_map, walks  


def log_dir():
    log_dir = FLAGS.base_log_dir + "/unsup-" 
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    print(log_dir)
    print('------------------------')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    all_nodes = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)  # size=256
        iter_num += 1
        outs_val = sess.run([model.outputs1], feed_dict=feed_dict_val)
        # ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                all_nodes.append(edge[0])
                val_embeddings.append(outs_val[-1][i, :])
                nodes.append(edge[0])
                seen.add(edge[0])

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)

    np.save(out_dir + '/emb_node.npy', np.array(nodes))
#    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    np.save(out_dir + '/embedding.npy',  val_embeddings)
#    with open(out_dir + name + mod + ".txt", "w") as fp:
#        print('------------------------------')
#        fp.write("\n".join(map(str,nodes)))
        
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss], feed_dict=feed_dict_val)
#    outs_val = sess.run([model.loss, model.ranks, model.mrr],
#                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def construct_placeholders():
    node_pred = FLAGS.node_pred
    if not node_pred:
        labels = 1
    else:
        labels = 40
    # Define placeholders shape
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, labels), name='labels'),
        'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
        'weight': tf.placeholder(tf.float32, shape=(None), name='weight'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,), name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

#%%
def train(train_data, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    # save id_map
    with open('./id_map.pkl', 'wb') as file:
        pickle.dump(id_map, file)
    labels = train_data[3]
    
    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            labels,
            placeholders, batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs = context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     concat=False,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = FLAGS.identity_dim,
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*FLAGS.dim_1,
                                       lr=FLAGS.learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # Train model
    
    train_shadow_mrr = None
    shadow_mrr = None

    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
#    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    val_adj_info = tf.assign(adj_info, minibatch.adj)
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle()
        total_steps = 0
        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        print(minibatch.end())
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
#            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            # outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, model.mrr, model.outputs1], feed_dict=feed_dict)
            outs = sess.run([merged, model.opt_op, model.loss, model.grad, model.node_preds, model.placeholders['labels'], model.outputs1, model.accuracy], feed_dict=feed_dict)
            train_cost = outs[2]
            grad = outs[3]
            node_pres = outs[4]
            label = outs[5]
            train_acc = outs[7]
#            train_tmrr = outs[5]
#            if train_shadow_mrr is None:
#                train_shadow_mrr = train_mrr#
#            else:
#                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                feed_dict_val, labels_val = minibatch.val_shuffle()
                outs_val = sess.run([model.loss, model.node_preds, model.placeholders['labels'], model.accuracy, model.predicted], feed_dict=feed_dict_val)
                accuracy = outs_val[3]
                true_value = outs_val[2][:10]
                predicted_value = outs_val[4][:10]
                loss = outs_val[0]

                 # Validation
#                sess.run(val_adj_info.op)
#                val_cost, ranks, val_mrr, duration  = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
#                sess.run(train_adj_info.op)
#                epoch_val_costs[-1] += val_cost
#            if shadow_mrr is None:
#                shadow_mrr = val_mrr
#            else:
#                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            # avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
            avg_time = time.time() - t

            if total_steps % FLAGS.print_every == 0:

                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      'train_acc=', "{:.5f}".format(train_acc),
#                      "train_mrr=", "{:.5f}".format(train_mrr), 
#                      "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
#                      "val_loss=", "{:.5f}".format(val_cost),
#                      "val_mrr=", "{:.5f}".format(val_mrr), 
#                      "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break
    
        print('val_accuracy : ' + str(accuracy) + ' val_loss : ' + (str(loss)))
        # print('true_value : ' + str(true_value.T))
        # print('predicted_value : ' + str(predicted_value.T))
    print("Optimization Finished!")
    all_vars = tf.trainable_variables()
    # save variable, https://blog.csdn.net/u012436149/article/details/56665612
    dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense_1_vars')
    # saver = tf.train.Saver(all_vars[5:])
    # save model
    saver = tf.train.Saver()
    saver.save(sess, "./model_output/model")  # save entire model/ session

    # print(all_vars[5:])
    # print(dense_vars)
    # saver.save(sess, "./var_output/model")

    # current path
    # print(os.path.abspath(os.getcwd()))
    # print(os.getcwd())

    if FLAGS.save_embeddings:
        sess.run(val_adj_info.op)

        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, 'author_venue_embedding')

#        if FLAGS.model == "n2v":
#            # stopping the gradient for the already trained nodes
#            train_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
#                    dtype=tf.int32)
#            test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']], 
#                    dtype=tf.int32)
#            update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
#            no_update_nodes = tf.nn.embedding_lookup(model.context_embeds,tf.squeeze(train_ids))
#            update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
#            no_update_nodes = tf.stop_gradient(tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
#            model.context_embeds = update_nodes + no_update_nodes
#            sess.run(model.context_embeds)
#
#            # run random walks
#            from graphsage.utils import run_random_walks
#            nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
#            start_time = time.time()
#            pairs = run_random_walks(G, nodes, num_walks=50)
#            walk_time = time.time() - start_time
#
#            test_minibatch = EdgeMinibatchIterator(G, 
#                id_map,
#                placeholders, batch_size=FLAGS.batch_size,
#                max_degree=FLAGS.max_degree, 
#                num_neg_samples=FLAGS.neg_sample_size,
#                context_pairs = pairs,
#                n2v_retrain=True,
#                fixed_n2v=True)
#            
#            start_time = time.time()
#            print("Doing test training for n2v.")
#            test_steps = 0
#            for epoch in range(FLAGS.n2v_test_epochs):
#                test_minibatch.shuffle()
#                while not test_minibatch.end():
#                    feed_dict = test_minibatch.next_minibatch_feed_dict()
#                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all, 
#                        model.mrr, model.outputs1], feed_dict=feed_dict)
#                    if test_steps % FLAGS.print_every == 0:
#                        print("Iter:", '%04d' % test_steps, 
#                              "train_loss=", "{:.5f}".format(outs[1]),
#                              "train_mrr=", "{:.5f}".format(outs[-2]))
#                    test_steps += 1
#            train_time = time.time() - start_time
#            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(), mod="-test")
#            print("Total time: ", train_time+walk_time)
#            print("Walk time: ", walk_time)
#            print("Train time: ", train_time)


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, load_walks=True)
    print("Done loading training data..")
    
    train(train_data)

if __name__ == '__main__':
    tf.app.run()

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
import sys
from dataloader import gen_edges
from tensorflow.python.platform import flags

#%%
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
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
flags.DEFINE_integer('neg_size', 100, 'default is 2')  # custom ones
flags.DEFINE_integer('time_step', 280, 'default is 280')
flags.DEFINE_integer('dim_1', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 50, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 4096, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 50, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_boolean('node_pred', False, 'Which task to perform')

# logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', 'embedding', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 500, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 4096, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 100, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10000, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def load_data(prefix, normalize=True, load_walks=None, time_step=None, draw_G=True, save_fig='graph', time=281):
    # assert time_step != None, "load_data-- time_step can't None!"

    # rolling到321時全部的node
    all_edges = gen_edges(322)[['head', 'tail']].values
    all_edges = np.unique(all_edges.flatten())

    # t時刻該有的edges
    all_edge = gen_edges(time+1)[['head', 'tail']].values
    # gen_edges(time, True)  # for testing purpose

    # 建一張空表有全部 node
    G = nx.DiGraph()
    G.add_edges_from(all_edge)
    # 補齊剩下缺少的 nodes
    t_nodes = np.unique(all_edge.flatten())
    missing_nodes = np.setdiff1d(all_edges, t_nodes)
    # 手動把缺少的加進G
    G.add_nodes_from(missing_nodes)

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

    np.save(out_dir + '/emb_node' + str(FLAGS.time_step) + '.npy', np.array(nodes))
#    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    np.save(out_dir + '/embedding' + str(FLAGS.time_step) + '.npy',  val_embeddings)
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

# %%
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
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    # 把舊的 model讀近來做 fine-tune
    # saver.restore(sess, "./model_output/model")
    
    # Train model
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

        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            # outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, model.mrr, model.outputs1], feed_dict=feed_dict)
            outs = sess.run([merged, model.opt_op, model.loss, model.grad, model.node_preds, model.placeholders['labels'], model.outputs1, model.accuracy], feed_dict=feed_dict)
            train_cost = outs[2]
            # grad = outs[3]
            # node_pres = outs[4]
            # label = outs[5]
            train_acc = outs[7]
#            train_tmrr = outs[5]

            if iter % FLAGS.validate_iter == 0:
                feed_dict_val, labels_val = minibatch.val_shuffle()
                outs_val = sess.run([model.loss, model.node_preds, model.placeholders['labels'], model.accuracy, model.predicted], feed_dict=feed_dict_val)
                accuracy = outs_val[3]
                # true_value = outs_val[2][:10]
                # predicted_value = outs_val[4][:10]
                loss = outs_val[0]

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            avg_time = time.time() - t

            if total_steps % FLAGS.print_every == 0:

                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      'train_pos_acc=', "{:.5f}".format(train_acc[0]),
                      'train_neg_acc=', "{:.5f}".format(train_acc[1]),
                      'train_overall_acc=', "{:.5f}".format(train_acc[2]),
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break
    
        print('val_pos_accuracy : ' + str(accuracy[0]) + ' val_loss : ' + (str(loss)))
        print('val_recall : ' + str(accuracy[4]) + ' val_precision : ' + (str(accuracy[3])))
        # print('true_value : ' + str(true_value.T))
        # print('predicted_value : ' + str(predicted_value.T))

    print("Optimization Finished!")
    all_vars = tf.trainable_variables()
    # save variable, https://blog.csdn.net/u012436149/article/details/56665612
    # dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense_1_vars')
    # saver = tf.train.Saver(all_vars[5:])

    # save model
    # saver = tf.train.Saver()
    directory = "./rolling_models/" + str(FLAGS.time_step)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # saver.save(sess, "./rolling_models/" + str(FLAGS.time_step) + "/model")  # save entire model/ session
    saver.save(sess, "./model_output/model")

    # print(all_vars[5:])
    # print(dense_vars)
    # saver.save(sess, "./var_output/model")

    # current path
    # print(os.path.abspath(os.getcwd()))
    # print(os.getcwd())

    if FLAGS.save_embeddings:
        sess.run(val_adj_info.op)
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, 'author_venue_embedding')

    # sess.close()
    # tf.reset_default_graph()


def main(argv=None):
    times = [284, 302, 307, 310, 318, 321]
    for t in times:
        FLAGS.time_step = t
        print("Loading training data..")
        train_data = load_data(FLAGS.train_prefix, load_walks=True, time=t)
        print("Done loading training data..")

        train(train_data)

if __name__ == '__main__':
    # tf.app.run()
    flags_passthrough = flags.FLAGS.flag_values_dict()
    main = main or sys.modules['__main__'].main
    main(flags_passthrough)


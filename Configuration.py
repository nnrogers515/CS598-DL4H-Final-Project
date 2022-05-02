#!/usr/bin/env python
# coding: utf-8

'''
Class Designed For Updating HyperParameters in the Code
Used Similarly in the Paper. Some parameters are unused as the original did not use them
'''

class Configuration:
    dataset = "data"
    data_path = "./%s" % dataset
    # checkpoint_dir = "checkpoint" # UNUSED
    # decay_rate = 0.95 # UNUSED
    # decay_step = 1000 # UNUSED
    n_topics = 50
    learning_rate = 0.00002
    vocab_size = 619
    n_stops = 22 
    lda_vocab_size = vocab_size - n_stops
    n_hidden = 200
    # n_layers = 2 # UNUSED
    projector_embed_dim = 100
    # generator_embed_dim = n_hidden # UNUSED
    # dropout = 1.0 # UNUSED
    # max_grad_norm = 1.0 #for gradient clipping # UNUSED
    grad_clip = 100
    total_epoch = 5
    epoch_size = 100
    batch_size = 1
    # init_scale = 0.075 # UNUSED
    threshold = 0.5 #probability cut-off for predicting label to be 1
    # forward_only = False #indicates whether we are in testing or training mode # UNUSED
    # log_dir = './logs' # UNUSED


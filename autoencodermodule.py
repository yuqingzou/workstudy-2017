# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:30:50 2017
http://learningtensorflow.com/lesson4/

@author: hannah.li
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class autoencodermodle(object):
    """A customer of autocodermodle. 
    Attributes:
        number_of_layer: number of hidden layers. @int
        inputfile: the location of the input file. @string
    """

    def __init__(self,number_of_laye= 2.0,inputfile='',unit_list=[10,5]):
        """Return a autoencoder object with default 2 layers and and also 
        initial the parameter*."""
        self.number_of_laye = number_of_laye
        self.inputfile = inputfile
        self.unit_list = unit_list
        
        ### desire output attribute###
        self.hidden_1_weight = None
        self.hidden_1_biases = None
        self.hidden_2_weight = None
        self.hidden_2_biases = None
        
                
    def session(self):
        ###read the csv file #####
        df = pd.read_csv(self.inputfile, usecols = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15'])
        print(self.number_of_laye,self.inputfile)
        
        
        #### set the Parameters ####
        learning_rate = 0.01
        training_epochs = 20
        #### batch_size to 10###
        batch_size = 10
        display_step = 1
        examples_to_show = 10
        
        # Network Parameters
        n_hidden_1 = self.unit_list[0] # 1st layer num features
        n_hidden_2 = self.unit_list[1] # 2nd layer num features
        n_input = 15 # Q15 data input (img shape: 15)
        
        
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, n_input])

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([n_input])),
        }


        # Building the encoder
        def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                           biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                           biases['encoder_b2']))
            return layer_2


        # Building the decoder
        def decoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                           biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                           biases['decoder_b2']))
            return layer_2

        # Construct model
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X
    
        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
        # Initializing the variables
        init = tf.initialize_all_variables()
    
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            total_batch = int(len(df)/batch_size)
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                s = 0
                n = 10
                for i in range(total_batch):
                    batch_xs = df[s:s+n]
            # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                    s=s+n
#==============================================================================
#                     # Display logs per epoch step
#                     if epoch % display_step == 0:
#                         print("Epoch:", '%04d' % (epoch+1),
#                               "cost=", "{:.9f}".format(c))
#==============================================================================

            print("Optimization Finished!")
            self.hidden_1_weight = sess.run(weights['encoder_h1'])
            self.hidden_1_biases = sess.run(biases['encoder_b1'])
            self.hidden_2_weight = sess.run(weights['encoder_h2'])
            self.hidden_2_biases = sess.run(biases['encoder_b1'])
            
            
    def print_weight(self):
        """ print weight (variable object) for all layer"""
        print('first layer weight \n')
        print(self.hidden_1_weight)
        print('second layer weight\n')
        print(self.hidden_2_weight)
                
    def print_biases(self):
        """ print biases (variable object) for all layer"""
        print('first layer biases \n')
        print(self.hidden_1_biases)
        print('second layer biases\n')
        print(self.hidden_2_biases) 

#==============================================================================
#     # Applying encode and decode over test set
#     encode_decode = sess.run(
#         y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
#     # Compare original images with their reconstructions
#     f, a = plt.subplots(2, 10, figsize=(10, 2))
#     for i in range(examples_to_show):
#         a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#         a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
#     f.show()
#     plt.draw()
#     plt.waitforbuttonpress()
#==============================================================================



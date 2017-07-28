"""
convolutional autoencoder module 

yuqing zou 

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class convolutional_autoencoder_module(object):
    """A customer of Convolutional_autoencoder. 
        input parameter can be input size and shape, also the weight and baise 
        for each conv_encoder and conv_decoder laye.
    """

    def __init__(self,input_shape= [-1, 4, 16, 1],inputfile='', w_e_conv1 = [3,3,1,16],
        w_e_conv2 = [2,2,16,32], b_e_conv1 =[16], b_e_conv2 = [32], 
        b_d_conv1 =[1], b_d_conv2 = [16], pad_mode = "SAME",
        strides = 1):   
        """Return a convolutional_module object with default shape and some
        initial the parameter."""
        self.inputfile = inputfile
        
        ###Conve layer shape parameter
        self.input_shape = input_shape
        self.w_e_conv1 = w_e_conv1
        self.w_e_conv2 = w_e_conv2
        self.b_e_conv1 = b_e_conv1
        self.b_e_conv2 = b_e_conv2
        self.w_d_conv1 = w_e_conv2
        self.w_d_conv2 = w_e_conv1
        self.b_d_conv1 = b_d_conv1
        self.b_d_conv2 = b_d_conv2
        self.pad_mode = pad_mode
        #===formular (input_size - Con_size +2pad(0))/strides
        
        
        ### desire output attribute###
        self.hidden_1_weight = None
        self.hidden_1_biases = None
        self.hidden_2_weight = None
        self.hidden_2_biases = None
        
        # Parameters
        learning_rate = 0.001
        training_iters = 200000
        
                
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = self.pad_mode)

    def deconv2d(self,x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 1, 1, 1], padding = self.pad_mode)
        print("here is the conv2d")
        print(tf.shape(x))
        print(tf.Print(x,[x]))
        print(tf.Print(W,[W]))
        print(tf.shape(W))
    def max_pool_2x2(self,x):
        _, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = self.pad_mode)
        pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = self.pad_mode)
        return pool, argmax
        
        
    def max_unpool_2x2(self,x, output_shape):
        out = tf.concat([x, tf.zeros_like(x)],3)
        out = tf.concat([out, tf.zeros_like(out)],2)
        out_size = output_shape
        return tf.reshape(out, out_size)


           
    def session(self):
        """ pass in the input file and run tW_e_conv1he module"""

        #network parameter
        n_input = 64
        training_epochs = 5000

        #reshape
        tf.reset_default_graph()        
        x = tf.placeholder(tf.float32, shape = [None, n_input])
        print(tf.shape(x)[0])
        x_origin = tf.reshape(x, [-1, 4, 16, 1])            
        
        
        
        # Store layers weight & bias4
        #weight sttructure  4 D tensor [fh,fw,fn',fn]
        #fh and fw are the height and width of the receptve field' 
        #fn' is the number of feature map in previous layer 
        weights = {
        # 5x5 conv, 1 input, 32 outputs
        'W_e_conv1': tf.Variable(tf.random_normal(self.w_e_conv1)),
        # 5x5 conv, 32 inputs, 64 outputs
        'W_e_conv2': tf.Variable(tf.random_normal(self.w_e_conv2)),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_d_conv1': tf.Variable(tf.random_normal(self.w_d_conv1)),
        # 1024 inputs, 10 outputs (class prediction)
        'W_d_conv2': tf.Variable(tf.random_normal(self.w_d_conv2))
        }
        
        #biases is simply shape as [fn] fature number of the current feature #
        biases = {
        'b_e_conv1': tf.Variable(tf.random_normal(self.b_e_conv1)),
        'b_e_conv2': tf.Variable(tf.random_normal(self.b_e_conv2)),
        'b_d_conv1': tf.Variable(tf.random_normal(self.b_d_conv1)),
        'b_d_conv2': tf.Variable(tf.random_normal(self.b_d_conv2))
        }
        
        #first convolutional encoder layer
        h_e_conv1 = tf.nn.relu(tf.add(self.conv2d(x_origin, weights['W_e_conv1']), biases['b_e_conv1']))
        h_e_pool1, argmax_e_pool1 = self.max_pool_2x2(h_e_conv1)
        
        #second convolutional encoder layer
        h_e_conv2 = tf.nn.relu(tf.add(self.conv2d(h_e_pool1, weights['W_e_conv2']), biases['b_e_conv2']))
        h_e_pool2, argmax_e_pool2 = self.max_pool_2x2(h_e_conv2)
        
        #print encoder shape here
        code_layer = h_e_pool2
        print("h_e_pool1 shape :%s" % h_e_pool1.get_shape())
        print("code layer shape : %s" % code_layer.get_shape())
        
        #frist deconvolutional decoder layer
        size_axis0 = math.ceil(self.input_shape[1]/2)
  
        size_axis1 = math.ceil(self.input_shape[2]/2)

        #shape shoud be (?,1,4,16)
        output_shape_d_conv1 = tf.stack([tf.shape(x)[0],math.ceil(size_axis0/2),math.ceil(size_axis1/2),self.b_d_conv2[0]])        
        print("output_shape_d_conv1 : %s" % output_shape_d_conv1.get_shape())
        print("code layer %s wdconv1%s outputshape%s" % (code_layer.get_shape(),weights['W_d_conv1'].get_shape(),output_shape_d_conv1.get_shape()))
        h_d_conv1 = tf.nn.sigmoid(self.deconv2d(code_layer, weights['W_d_conv1'], output_shape_d_conv1))
        print("h_d_conv1 : %s" %h_d_conv1.get_shape())
        
        #first max_unpooling layer
        output_shape_d_pool1 = tf.stack([tf.shape(x)[0],size_axis0, size_axis1,self.b_d_conv2[0]])
        print("output_shape_d_pool1 : %s" % output_shape_d_pool1.get_shape())
        h_d_pool1 = self.max_unpool_2x2(h_d_conv1, output_shape_d_pool1)
        print("h_d_pool1 shape :%s" % h_d_pool1.get_shape())
   
        #second deconvoluntional decoder layer
        output_shape_d_conv2 = tf.stack([tf.shape(x)[0],size_axis0,size_axis1, self.input_shape[3]])
        h_d_conv2 = tf.nn.sigmoid(self.deconv2d(h_d_pool1, weights['W_d_conv2'], output_shape_d_conv2))
        
        #second max_unpooling layer 
        output_shape_d_pool2 = tf.stack([tf.shape(x)[0],self.input_shape[1],self.input_shape[2],self.input_shape[3]])
        h_d_pool2 = self.max_unpool_2x2(h_d_conv2, output_shape_d_pool2)
        
        #print reconstruct layer shape here
        x_reconstruct = h_d_pool2
        print("reconstruct layer shape : %s" % x_reconstruct.get_shape())
        
        #build the cost function here        
        cost = tf.reduce_mean(tf.pow(x_reconstruct - x_origin, 2))
        optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
        
        
        
        #input data here
        filename_queue = tf.train.string_input_producer(['./Q_15_train_mw.csv'])

        reader = tf.TextLineReader(skip_header_lines=1)
        _, csv_row = reader.read_up_to(filename_queue,4)
        record_default = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[""],[0.0],[0.0]]
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,pep,fm,label = tf.decode_csv(csv_row, record_defaults=record_default)
        features = tf.stack([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,label]) 
        
        
        #traning 
        sess = tf.InteractiveSession()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(training_epochs):
            example, labels = sess.run([features,label])
            example = example.transpose()
            example = example.reshape(1,64)
            if epoch < 1500:
                if epoch%100 == 0:
                    print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:example})))
            else:
                if epoch%1000 == 0: 
                    print("step %d, loss %g"%(epoch, cost.eval(feed_dict={x:example})))
            optimizer.run(feed_dict={x: example})
        
        #print("final loss %g" % cost.eval(feed_dict={x: mnist.test.images}))
        coord.request_stop()
        coord.join(threads)

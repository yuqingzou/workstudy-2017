# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:36:41 2017

@author: hannah.li
"""
import sys
import tensorflow as tf
import pandas as pd
from autoencodermodule import autoencodermodle


####flages####
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('md', 'autoencoder',
                       " choose ml module here ")
tf.app.flags.DEFINE_string('train_dir', './Q_15_train_mw.csv',
                       " Directory where the test ")
tf.app.flags.DEFINE_integer('max_layers', 3,
                        """ Number of layers for autoencoder. """)
tf.app.flags.DEFINE_string('later_set', '10 5 10',
                        """ set the number of layers """)

####main#####
def main():
  if FLAGS.md == 'autoencoder':
      ae = autoencodermodle(FLAGS.max_layers,FLAGS.train_dir)
      ae.session()
      ae.print_weight()
      ae.print_biases()
      sys.exit()
      
if __name__ == "__main__":
    main()

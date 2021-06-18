# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:30:29 2021

@author: giuli
"""

import tensorflow as tf
from tensorflow import keras
import math

class LearningRateScreening(keras.callbacks.Callback):
    def __init__(self, stop_factor = 4, max_lr = 10e9, n_batch_updates = 1000):
        super(LearningRateScreening, self).__init__()
        
        # initializing quantities
        # - learning rate list
        # - loss list
        # - batch count
        self.lr_list = []
        self.loss_list = []
        self.batch_num = 0
        
        # initialiting quantities
        # for lr exponential variation
        self.max_lr = max_lr
        self.nbatch_updates=n_batch_updates
        
        # initialize variables to 
        # control stop condition
        self.stop_factor = stop_factor
        self.best_loss = 1e9
        
    def on_batch_end(self, batch, logs):
        
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        
        # Get the current learning rate from model's optimizer.
        # store it into the list
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(" - learning rate: {}".format(lr))
        self.lr_list.append(lr)
        
        # Increase the batch number
        self.batch_num += 1 
        
        # get the loss value and store it into the list
        l = logs['loss']
        current_loss = l
        self.loss_list.append(l)
               
        # determine the value of the loss at which
        # training should be stopped
        loss_stop =  self.stop_factor * self.best_loss
        
        # stop if loss is greater than stop
        if self.batch_num > 1 and current_loss>loss_stop:
            self.model.stop_training = True
            return
        
        # else set current loss as best
        if self.batch_num==1 and current_loss<self.best_loss:
            self.best_loss=current_loss
        
        # Update the new learning rate 
        # based on an exponential growth
        # basic formula: x(t)=ab**(t) 
        # with:
        # - x(0) = a = start_lr 
        # - x(nbatch_updates) = end_lr
        # invert to obtain the desired growth factor
        f=(self.max_lr/self.lr_list[0])**(1/self.nbatch_updates)
        new_lr = lr*f
        
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        #print("\nEpoch %05d: Learning rate is %6.4f." % (batch, new_lr)) 

    
    
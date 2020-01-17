#!/usr/bin/env python
# coding: utf-8

import numpy as np 
def load_data( ):
    X_train = None 
    X_test = None 
    Y_train = None 
    Y_test = None
    
    ################################################################################################
    #Generate training and test data splits from input data 
    ################################################################################################
    #This file holds BERT embedding representation for 20News group , File Train_Data_Embeddings hold code for generating these embeddings
    X_train = np.genfromtxt('./20newsgroup_BERT_Embedding.txt') 
    Y_train = np.genfromtxt('./20newsgroup_Train_Labels.txt')
    
    X_test  = np.genfromtxt('./20newsgroup_BERT_Embedding_test.txt') 
    Y_test  = np.genfromtxt('./20newsgroup_Test_Labels.txt')
    
    return X_train, X_test, Y_train, Y_test
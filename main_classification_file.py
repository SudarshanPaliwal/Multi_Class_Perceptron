#!/usr/bin/env python
# coding: utf-8

"""
Multi class perceptron based on One vs All and Winner takes all , implemented using object oriented approach by creating class for
binary classifier with fit and predict methods . Multi class perecptron class creates binary classifiers equal to number of 
class labels. Look at file Data_Loader for loading data and file multi_class_perceptron for classes 
"""

import numpy as np
from Data_Loader import load_data
from multi_class_perceptron import multi_class_perceptron
from sklearn.metrics import accuracy_score # Only to report accuracy of model 
import pickle                            # For Saving model 


#Load Train and Test Data set 
X_train, X_test, Y_train, Y_test = load_data( )
#Number of unique class labels  which is also the number of classifiers we will train 
unique_labels   = np.unique([Y_train])
num_classifiers = unique_labels.size

#Train the model - Multi class perceptron from scratch without using any libraries Check file multi_class_perceptro for implementation
model = multi_class_perceptron(num_classifiers,X_train.shape[1],unique_labels,20000)

model.fit(X_train, Y_train)

#save the model 
filename = 'trained_model.pkl'
pickle.dump(model, open(filename, 'wb'))

#Run predictions on test data 
print('Start of Prediction on Test Data')
y_predicted = model.predict(X_test)
print('End of Prediction on Test Data')

print('Test Accuracy of model is ',accuracy_score(Y_test, y_predicted))   

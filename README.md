# Multi_Class_Perceptron
Multi Class Perceptron Model ( One Vs All )

Files : 
main_classification_file.py  - This is main file to run 
This file contains code to train multi class perceptron using one Vs All and Winner takes all approach for prediction . This file instantiates multiclassperceptron which in turn instantiates binaryclass perceptron for every class label . 

Data_Loader.py             - Load training and test data from files Train_Data_Embeddings.py and Test_Data_Embeddings.py

multi_class_perceptron.py  - Contains class and logic for training and predicting using numpy . 

Train_Data_Embeddings.py   - Code writen to Generate BERT embeddings for training data in file Train_Data_Embeddings.py

Test_Data_Embeddings.py    - Code writen Generate BERT embeddings for test data in file Test_Data_Embeddings.py

trained_model_500.pkl      - Trained model for 500 iterations 

Results_500_Iter.JPG       - Results on test data 


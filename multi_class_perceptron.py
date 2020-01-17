import numpy as np

import numpy as np

class perceptron(object):
    
    
    """
       This class creates perceptron for binary classifier providing fit and predcit method 
    """   
    def __init__(self, input_dim,max_iter):
        self.dims = input_dim
        self.w = np.zeros(self.dims + 1) # One xtra for bias 
        self.max_iter = max_iter
        
    def fit(self,X,y):
        ones = np.ones(X.shape[0]) 
        X = np.insert(X,0,ones,1) #Add ones to data for bias w0
        has_converged = False 
        iterations = 0 
        while has_converged == False and iterations < self.max_iter:      #Loop through all samples until it has converged 
        #for k in range(self.max_iter):
            has_converged = True          #Default it to True , it will set to false when even on sample is misclassifeid  
            for i in range(X.shape[0]): #Loop through all samples  
                if y[i] * np.dot(self.w, X[i]) <= 0 : # incorrect examples when y[i] and wx has opposite sign
                    has_converged = False
                    self.w = self.w + y[i]*X[i]
            iterations = iterations + 1         
        print(iterations)
        
    def predict(self,X):
        ones = np.ones(X.shape[0]) 
        X = np.insert(X,0,ones,1) #Add ones to data for bias w0        
        return np.dot(X,self.w)   #Since I am using one Vs All and Winner takes all I will store values of wTx and take the which has max value for classifier 

class multi_class_perceptron(object):
    
    """
       This class creates multiple perceptron obecjst for each label 
    """
    def __init__(self, num_classifiers,num_dims,unique_labels,max_iters):
        
        self.tot_binary_classifiers = num_classifiers #Total Binary classifiers - One per label 
        self.num_dims               = num_dims        #Number of dimensions 
        self.unique_labels          = unique_labels   #Number of unique labels - Used to create target labels +1 and -1 ( One Vs All )
        self.classifiers            = np.empty(num_classifiers, dtype=object) 
        for i in range(int(num_classifiers)):
            self.classifiers[i] = perceptron(num_dims,max_iters)         

    
    def fit(self,X,y):
        
        for i in range(int(self.tot_binary_classifiers)):  #Number of classifiers equal to number of unique class labels             
            classifier_y = np.copy(y)             
            #Set value in target to +1 for each class label and train binary classifier for each of classes  
            updated_y = np.where(classifier_y == self.unique_labels[i],+1,-1)
            print('Start of Training Binary Classifier ' , i+1 )
            self.classifiers[i].fit(X,updated_y)
            print('End of Training Binary Classifier ' , i+1 )
        
    def predict(self, X):
        """
         Predicion for One Vs All is based on winner takes all , predict function of binary classifier returns w^Tx value and we predict class label of test data to be the one with highest value of w^Tx
        """
        Y = np.empty([X.shape[0],self.tot_binary_classifiers]) # Target matrix to hold value of predictions for each binary classifier 
        for i in range(int(self.tot_binary_classifiers)):  #Predict value for complete test data matrix at once for each classifier         
            Y[:,i] = self.classifiers[i].predict(X) # It returns a vector for predicted value for each data sample in matrix                     
#Get the column index with max value of w^Tx for given data point(which is per row) and find class label for that index 
        return self.unique_labels[np.argmax(Y, axis=1)]    
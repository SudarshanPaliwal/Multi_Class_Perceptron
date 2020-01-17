#!/usr/bin/env python
# coding: utf-8

# In[17]:


from sklearn.datasets import fetch_20newsgroups
newsgroups_test = fetch_20newsgroups(subset='test')
np.savetxt('20newsgroup_Test_Labels.txt',newsgroups_test.target) #Save Test target labels 


# In[ ]:


import torch   #Used only for BERT embedding represenations 
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM #Used only for BERT embedding represenations
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval() 


# In[ ]:


#Generate BERT word embeddings for all the file data by looping over data 

embedding_vector = np.empty([len(newsgroups_test.data),model.embeddings.word_embeddings.embedding_dim])

for i in range(len(newsgroups_test.data)) :
    
    #Split each sentence and generate word embedding and mean for all sentences in file 
    data_no_newline = newsgroups_test.data[i].replace('\n','') #Replace new line characters 
    Sentences = data_no_newline.split('.')
    sentence_embedding_vector = np.empty([len(Sentences),model.embeddings.word_embeddings.embedding_dim])
    for j in range (len(Sentences)):
        text = Sentences[j]
        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)   

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Predict hidden states features for each layer
        if tokens_tensor.shape[1] > 511 :
            tokens_tensor = tokens_tensor[:,0:512] 
            segments_tensors = segments_tensors[:,0:512]             
        with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)

        #Create embedding for whole sentence using average in last layer for all tokes using     
        #the average of all  token vectors.      
        sentence_embedding_vector[j] = np.array(encoded_layers[11][0]).mean(axis = 0)    
    embedding_vector[i] = sentence_embedding_vector.mean(axis = 0)
np.savetxt('20newsgroup_BERT_Embedding_test.txt',embedding_vector)        


#!/usr/bin/env python
# coding: utf-8

# In[82]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[9]:


# Import data
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')


# In[10]:


# Formatting data & adding classification
fake['class'] = 'fake'
true['class'] = 'true'
factCheck = pd.concat([fake,true]).reset_index(drop=True)


# In[11]:


factCheck.head()


# In[12]:


# Shorten dataset
factCheck = factCheck.sample(frac=1)
factCheck = factCheck[1:1001]


# In[46]:


# Corpus
corpus = factCheck[['text']].values.tolist()


# In[57]:


bag_of_words = []
for x,y in enumerate(corpus):
    z = str(y[0])
    bag_of_words.append(z)


# In[97]:


vectorizer = CountVectorizer(analyzer = "word", 
                             lowercase=True, 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 5000,
                             token_pattern='[a-zA-Z0-9$&+,:;=?@#|<>^*()%!-]+')


# In[99]:


word_matrix = vectorizer.fit_transform(bag_of_words)
print(word_matrix.todense())


# In[101]:


print(word_matrix.shape)


# In[102]:


vocab = vectorizer.vocabulary_
print(vocab)


# In[103]:


tokens = vectorizer.get_feature_names()
print(tokens)


# In[104]:


docNames = ['Doc{:d}'.format(idx) for idx, _ in enumerate(word_matrix)]
df = pd.DataFrame(data=word_matrix.toarray(), index = docNames,
                 columns = tokens)


# In[ ]:


df['class'] = ['0']


# In[127]:


df['class']


# In[ ]:





# In[ ]:





# In[76]:


target_values = factCheck[['class']].values.tolist()
targets = []
for x,y in enumerate(target_values):
    a = str(y[0])
    targets.append(a)


# In[78]:


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, targets, test_size = 0.20, random_state = 42)


# In[91]:


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


# In[92]:


X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[93]:


# Model
model = GaussianNB()
model.fit(X_train, y_train)


# In[85]:





# In[ ]:





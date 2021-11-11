#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Import libraries
import pandas as pd

from textblob.classifiers import NaiveBayesClassifier


# In[36]:


# Import data
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')


# In[37]:


# Formatting data & adding classification
fake['class'] = 'fake'
true['class'] = 'true'
factCheck = pd.concat([fake,true]).reset_index(drop=True)
factCheck.head()


# In[38]:


factCheck = factCheck.sample(frac=1)
model_data = factCheck[['text', 'class']]
model_data = model_data.values.tolist()


# In[39]:


train = model_data[1:800]
test = model_data[801:1000]


# In[42]:


model = NaiveBayesClassifier(train)


# In[43]:


model.accuracy(test)


# In[ ]:


# Test classification
model.classify('An apple is a fruit')


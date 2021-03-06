#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def fact_check(string):
    import pandas as pd
    from textblob.classifiers import NaiveBayesClassifier
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    fake['class'] = 'fake'
    true['class'] = 'true'
    factCheck = pd.concat([fake,true]).reset_index(drop=True)
    factCheck = factCheck.sample(frac=1)
    model_data = factCheck[['text', 'class']]
    model_data = model_data.values.tolist()
    train = model_data[1:800]
    test = model_data[801:1000]
    model = NaiveBayesClassifier(train)
    result = model.classify(string)
    return result


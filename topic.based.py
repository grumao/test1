#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import

import nltk
nltk.download('stopwords')

import spacy
import sys
get_ipython().system('{sys.executable} -m spacy download en')


# In[7]:


#Import

import re
import numpy as np
import pandas as pd
from pprint import pprint


# In[28]:


#Import

import sys
get_ipython().system('{sys.executable} -m pip install gensim --no-warn-script-location')
import gensim


# In[29]:


#Import

import gensim.corpora as corpora


# In[30]:


#Import

from gensim.utils import simple_preprocess


# In[31]:


#Import

from gensim.models import CoherenceModel


# In[35]:


#Import

import sys
get_ipython().system('{sys.executable} -m pip install pyLDAvis --user')
import pyLDAvis
import sys
get_ipython().system('{sys.executable} -m pip install pyLDAvis.gensim ')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


#Import & Stop Words

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# In[37]:


#######getting string from website

#df = pd.read_html('http:54.166.246.24:8000/')
#print(df.target_names.unique())
#df.head()
def topic_based():

df = [
    'There is a game on tonight that is very popular',
    'Blue cheese is my favorite food',
    'My middle name is boring',
    'Time has gone so fast, I miss my college friends',
    'COVID epically shaped the way todays world works'
]


# In[40]:


#clean data

#remove newline characters 

df = [re.sub('\s+',' ', sent)for sent in df]

#remove quotes
df = [re.sub("\'"," ", sent)for sent in df]

pprint(df[:1])


# In[43]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))
        
df_words = list(sent_to_words(df))

print(df_words[:1])


# In[44]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(df_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[df_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[df_words[0]]])


# In[45]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[48]:


#Call functions in order

# Remove Stop Words
df_words_nostops = remove_stopwords(df_words)

# Form Bigrams
df_words_bigrams = make_bigrams(df_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
df_lemmatized = lemmatization(df_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(df_lemmatized[:1])


# In[49]:


# Create Dictionary
id2word = corpora.Dictionary(df_lemmatized)

# Create Corpus
texts = df_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# In[50]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[51]:


#View the topics in LDA model

#view the keywords for each topic and the weightage(importance) of each keyword 

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[ ]:


#Used with accordance to the following link up until 'view topics'

#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/


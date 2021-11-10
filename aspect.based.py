#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
get_ipython().system('{sys.executable} -m pip install spacy')


# In[8]:


import spacy
import sys
get_ipython().system('{sys.executable} -m spacy download en_core_web_sm')
nlp = spacy.load("en_core_web_sm")


# In[9]:


#This is where we are supposed to read in string from link; "sentence" is just a test to prove the module is working

#import requests
#from bs4 import BeautifulSoup
#import pandas as pd

#string = []

#for strings in range (1,101):
    #save data
#website = 'http://54.166.246.24:8000/'
#quest = requests.get(website)
#cont = quest.content
#source = BeautifulSoup(cont, 'html.parser')
#text = source.find_all(text=True) 
    #name
#nameLookUp = 'Enter you String'
#strings = [idx for idx in text if idx.lower().startswith(nameLookUp.lower())]
#string.insert(0, strings[0][6:])
    
    
#sentences = pd.DataFrame([string]).T
#print(coData)    

sentences = [
  'The food we had yesterday was delicious',
  'My time in Italy was very enjoyable',
  'I found the meal to be tasty',
  'The internet was slow.',
  'Our experience was suboptimal'
]


# In[10]:


for sentence in sentences:
  doc = nlp(sentence)
  for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
      token.pos_,[child for child in token.children])


# In[11]:


for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  for token in doc:
    if token.pos_ == 'ADJ':
      descriptive_term = token
  print(sentence)
  print(descriptive_term)


# In[12]:


for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  for token in doc:
    if token.pos_ == 'ADJ':
      prepend = ''
      for child in token.children:
        if child.pos_ != 'ADV':
          continue
        prepend += child.text + ' '
      descriptive_term = prepend + token.text
  print(sentence)
  print(descriptive_term)


# In[13]:


aspects = []
for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  target = ''
  for token in doc:
    if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
      target = token.text
    if token.pos_ == 'ADJ':
      prepend = ''
      for child in token.children:
        if child.pos_ != 'ADV':
          continue
        prepend += child.text + ' '
      descriptive_term = prepend + token.text
  aspects.append({'aspect': target,
    'description': descriptive_term})
print(aspects)


# In[14]:


import sys
get_ipython().system('{sys.executable} -m pip install textblob')
import textblob


# In[15]:


from textblob import TextBlob
for aspect in aspects:
  aspect['sentiment'] = TextBlob(aspect['description']).sentiment
print(aspects)


# In[16]:


import sys
get_ipython().system('{sys.executable} -m textblob.download_corpora')


# In[17]:


from textblob.classifiers import NaiveBayesClassifier

train = [
  ('Slow internet.', 'negative'),
  ('Delicious food', 'positive'),
  ('Suboptimal experience', 'negative'),
  ('Very enjoyable time', 'positive'),
  ('delicious food.', 'neg')
]
cl = NaiveBayesClassifier(train)
# And then we try to classify some sample sentences.
blob = TextBlob("Delicious food. Very Slow internet. Suboptimal experience. Enjoyable food.", classifier=cl)
for s in blob.sentences:
  print(s)
  print(s.classify())


# In[ ]:


#Sources 

#https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a
#https://towardsdatascience.com/aspect-based-sentiment-analysis-using-spacy-textblob-4c8de3e0d2b9


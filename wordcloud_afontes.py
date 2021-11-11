#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[2]:


# Example text for sake of writing code:
example_text = '''NLP and NLU were big buzzwords in the tech community just a year ago.
Their use in a wide array of business applications stems significantly from the
incredible amount of work that has been put in to making NLP libraries available
and accessible. NLP has seen some decline (or maybe “less ascent”) as the big
chatbot boom of 2015 subsides, but the existing libraries are here to stay and
will remain useful in tasks like automated document summarizing and human
machine interfaces. Largely we are getting away from the “hype” and into the
“useful work” part of a technology’s life cycle.'''


# In[3]:


# Generate wordcloud
# Create wordcloud
wc = WordCloud(background_color='white', width = 300, height=300, margin=2).generate(example_text)


# In[8]:


plt.figure(figsize=(8,8), facecolor = 'white')
plt.imshow(wc)
plt.axis('off')
plt.tight_layout(pad=2)
plt.savefig('word_cloud.png')


# In[5]:





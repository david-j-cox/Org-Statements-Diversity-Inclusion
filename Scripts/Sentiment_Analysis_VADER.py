#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
    David J. Cox, PhD, MSB, BCBA-D
    https://www.researchgate.net/profile/David_Cox26
    twitter: @davidjcox_
    LinkedIn: https://www.linkedin.com/in/coxdavidj/
    Website: https://davidjcox.xyz
"""
#Set current working directory to the folder that contains your data.
import os
import pandas as pd
import numpy as np
import sys
import re, string, unicodedata
import matplotlib.pyplot as plt
sys.path.append('/Users/davidjcox/Dropbox/Coding/Local Python Modules/')

# Set path to data
os.chdir('/Users/davidjcox/Dropbox/Projects/CurrentProjectManuscripts/Empirical/PersonalFun/Org_Statements_Diversity/Org-Statements-Diversity-Inclusion/Data')

# Change settings to view all columns of data
pd.set_option('display.max_columns', None)

#%% Import data.
raw_data = pd.read_csv('all_data.csv').drop(['Unnamed: 0'], axis=1)
data = raw_data.copy()
data

#%% DATA PRE-PROCESSING 
# Pre-process our data. Goal is to have:
#       (1) Single list where each item in the list is the raw string of the narrative for that participant. 
#       (2) List of lists with one list per subject, and each item in list is a sentence from their narrative. 
#       (3) List of lists with one list per subject, and each item in the list is a clean* word from their narrative. 
#       (4) Single list with all of the cleaned vocab for the entire group. 
#       (5) Single list of the vocabulary used throughout all narratives (i.e., omitting all redundancies from (4)).
#       (6) Single list where each item in the list is a string of the cleaned narrative for that participant. 
#       (7) Single list where each item in the list is a string of the participant narratives with only clean words. 
#       ----------------------
#   List names for above:
#       (1) narratives
#       (2) narratives_sent_tokenized
#       (3) clean_words_tokenized
#       (4) narratives_word_list
#       (5) narrative_vocab 
#       (6) narr_as_string
#       (7) clean_ind_narr
#       --------------------------------------------------
#       * Clean = punctuation and stop words removed. 

#%% Start with (1) narratives:
#   Single list where each item in the list is the raw string of the narrative for that participant. 
narratives = data['body_text'] # Create a list of the narratives.

#%% Next we'll get (2), narratives_sent_tokenized:
#     List of lists with one list per subject, and each item in list is a sentence from their narrative. 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download() # Need to download the 'punkt' model from nltk if not already on computer.

# Make some empty lists we'll store our data in. 
lower_narratives = []
narratives_sent_tokenized = []
count=0
# Make lowercase all words in the narratives. Store in lower_narratives list. 
for each_narr in narratives:
    lower = each_narr.lower()
    lower = lower.replace("/", '')
    lower = lower.replace("\\", '')
    lower = lower.replace("_", ' ')
    lower_narratives.append(lower)
len(lower_narratives) # Should still have 1141 narratives. 
lower_narratives[:2]  # Check out first few to make sure everything looks okay. 

# Sentence tokenize the narratives. Store in narratives_sent_tokenized list. 
for each_narr in lower_narratives:
    sent_tokens = nltk.sent_tokenize(each_narr)
    narratives_sent_tokenized.append(sent_tokens)
print(len(narratives_sent_tokenized)) # Should still have 1141 narratives. 
narratives_sent_tokenized[::500] # Check out every 500th narrative to make sure it looks okay. 

#%% Next, we'll get (3), clean_words_tokenized:
#     List of lists with one list per subject, and each item in the list is a clean* word from their narrative.

# Some empty lists we'll need to store data. 
stem_narratives = []
clean_words_tokenized = []
narratives_word_tokenized = []

# Word tokenize the narratives. Store in narratives_word_tokenized list. 
for list in lower_narratives:
    word_tokens = nltk.word_tokenize(list)
    narratives_word_tokenized.append(word_tokens)
len(narratives_word_tokenized)                          # Should still have 1141 items. 
narratives_word_tokenized[:1]                          # Check to make sure the last two look okay. 


# Convert each word to its root (i.e., lemmatize). 
from nltk.stem import WordNetLemmatizer
wnl = nltk.WordNetLemmatizer()
for list in narratives_word_tokenized:
    temp_list = []
    for word in list:
        words_stemmed = wnl.lemmatize(word) # Noun
        words_stemmed = wnl.lemmatize(word, pos='v') # Verb
        temp_list.append(words_stemmed)
    stem_narratives.append(temp_list)
len(stem_narratives)                                    # Should still have 1141 items. 
stem_narratives[:1]                                    # Check last two and compare to narratives_word_tokenized

# Some additional punctuation characters. 
punctuation = [",", ".", "''", "' '", "\"", "!", "?", '-', '``', ':', ';', \
               "'s", "...", "'d", '(', ')', '=', "'", "#", "$", "%", "&", '_', \
               "<", "=v=", ">", "@", "[", "]", "^_^", '{', '}', "\"", '/', "\\\\", \
               "n't", "'ll", "'m", '*', '..', "\"links:\"", "[001]", "[002]", \
               "[003]", "<b>", "\"buttons\"", "\\r", "\\n", "\\\"", "\""]  # Define list of punctuation to remove. 

# Remove all punctuation, any sticky contraction elements, and stopwords from stem_narratives list.
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
a_range = [0, 1, 2, 3, 4, 5]

for i in a_range:
    for list in stem_narratives:
        for word in list:
            if word == "'ve":
                list.remove(word)
            if word in punctuation:
                list.remove(word)
            if word in stop_words:
                list.remove(word)

# Put this cleaned list into it's own list so we don't mess it up. 
clean_words_tokenized = stem_narratives
len(clean_words_tokenized) # Should still have 5044 items
clean_words_tokenized[::200] # Check look the same. 

#%% Next, we'll get (4) narratives_word_list:
#     (4) Single list with all of the cleaned vocab for the entire group. 
# ======================================================================================================================================
# Create empty list where we'll store our data. 
narratives_word_list = []

# Iterate over each list and add the word to the list we just created. 
for list in stem_narratives:
    for word in list:
        narratives_word_list.append(word)

narr_all_words = ' '.join(narratives_word_list)

len(narratives_word_list) # Should be 4038 total words.

#%% Next we'll get (5) narrative_vocab:
#   Single list of the vocabulary used throughout all narratives (i.e., omitting all redundancies from (4)).

# Create empty list where we'll store our data. 
narrative_vocab = []

# Iterate over narratives_word_list and only append to narrative_vocab if the word is not in there already. 
for word in narratives_word_list:
    if word not in narrative_vocab:
        narrative_vocab.append(word)
print("Number of words in vocab:", len(narrative_vocab)) # Should be 1373 unique words in our vocab. 
sorted_vocab = sorted(narrative_vocab)
unique_words = np.unique(sorted_vocab) # Look at every 100th word in the vocab set. 

#%% Next, we'll get (6) narr_as_string:
#   Single item of all narratives as a single string of the cleaned narratives. 

# Create empty list where we'll store our data. 
narr_as_string = []

# Join all of the words into single string. 
narr_as_string = ' '.join(narratives_word_list)
print("Number of characters total:", len(narr_as_string)) # Should be 31,973 characters in this string. 
narr_as_string[:198] # Look at the first 300 characters of this string. 

#%% Finally, we'll get (7) clean_ind_narr:
#     Single list where each item in the list is a string of the participant narratives with only clean words. 
clean_ind_narr = []
for list in clean_words_tokenized:
    sub_clean_narr = ' '.join(list)
    clean_ind_narr.append(sub_clean_narr)
data['cleaned_sentences'] = clean_ind_narr
print("Number of total statements", len(clean_ind_narr))
print(clean_ind_narr[::500])

#%% ===========================================================================
############################## LIST CREATION COMPLETE #########################
# =============================================================================
narratives                  # Single list where each item in the list is the raw string of the narrative for that participant. 
narratives_sent_tokenized   # List of lists with one list per subject, and each item in list is a sentence from their narrative. 
clean_words_tokenized       # List of lists with one list per subject, and each item in the list is a clean word from their narrative. 
narratives_word_list        # Single list with all of the cleaned words for the entire group. 
narr_all_words              # Single item of all the cleaned words for entire group as single string
narrative_vocab             # Single list of the vocabulary used throughout all narratives (i.e., omitting all redundancies from (4)).
narr_as_string              # Single item of all narratives as a string of the cleaned narratives. 
clean_ind_narr              # Single list where each item in the list is a string of the participant narratives with only clean words.

#%%        
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#%% All words. 
wordcloud = WordCloud(width=500, height=500, background_color='white').generate(narr_all_words) 
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

#%% Just top 50 words. 
wordcloud = WordCloud(width=500, height=500, background_color='white', max_words=50).generate(narr_all_words) 
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

#%% Sentiment Analysis on the raw input
from nltk.sentiment.vader import SentimentIntensityAnalyzer
raw_sentiment_score = []

for sentence in data['body_text']:
    ss = SentimentIntensityAnalyzer().polarity_scores(sentence)
    raw_sentiment_score.append(ss)

raw_sent_df = pd.DataFrame(raw_sentiment_score)
raw_sent_df = raw_sent_df.rename(columns={'neg':'raw_neg', 'neu':'raw_neu', \
                                  'pos':'raw_pos', 'compound':'raw_compound'})
raw_sent_df

#%% Sentiment Analysis on the lemmatized and cleaned sentences
lemmed_sentiment_score = []

for sentence in clean_ind_narr:
    ss = SentimentIntensityAnalyzer().polarity_scores(sentence)
    lemmed_sentiment_score.append(ss)

lemmed_sent_df = pd.DataFrame(lemmed_sentiment_score)
lemmed_sent_df = lemmed_sent_df.rename(columns={'neg':'cleaned_neg', 'neu':'cleaned_neu', \
                                  'pos':'cleaned_pos', 'compound':'cleaned_compound'})
lemmed_sent_df

#%% Add the above to the data df
data = pd.concat([data, raw_sent_df, lemmed_sent_df], axis=1)
data.to_csv("all_data.csv")
data[:]

#%% Doughnut plot of raw sentiment
# Data
neg_sum = data['raw_neg'].sum()
neu_sum = data['raw_neu'].sum()
pos_sum = data['raw_pos'].sum()
tot = neg_sum + neu_sum + pos_sum
sent_prop = [round((neg_sum/tot)*100, 2), round((neu_sum/tot)*100, 2), round((pos_sum/tot)*100, 2)]

#%% Doughnut plot
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

labels = ["Negative","Neutral","Positive"]

wedges, texts = ax.pie(sent_prop, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    if labels[i] == 'Negative':
        ax.annotate(labels[i]+' '+str(sent_prop[i])+'%', xy=(x, y), xytext=(1.35*np.sign(x), y),\
                    horizontalalignment=horizontalalignment, **kw)
    else:
        ax.annotate(labels[i]+' '+str(sent_prop[i])+'%', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),\
                    horizontalalignment=horizontalalignment, **kw)
        
ax.set_title("Overall Raw Sentiment")
plt.show()

#%% Doughnut plot of cleaned
# Data
neg_sum = data['cleaned_neg'].sum()
neu_sum = data['cleaned_neu'].sum()
pos_sum = data['cleaned_pos'].sum()
tot = neg_sum + neu_sum + pos_sum
sent_prop = [round((neg_sum/tot)*100, 2), round((neu_sum/tot)*100, 2), round((pos_sum/tot)*100, 2)]

#%% Doughnut plot
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

labels = ["Negative","Neutral","Positive"]

wedges, texts = ax.pie(sent_prop, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    if labels[i] == 'Negative':
        ax.annotate(labels[i]+' '+str(sent_prop[i])+'%', xy=(x, y), xytext=(1.35*np.sign(x), y),\
                    horizontalalignment=horizontalalignment, **kw)
    else:
        ax.annotate(labels[i]+' '+str(sent_prop[i])+'%', xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),\
                    horizontalalignment=horizontalalignment, **kw)
        
ax.set_title("Overall Cleaned Sentiment")
plt.show()

#%% Hist of raw compound scores
def hist_plot(df, title):
    fig = plt.subplots(figsize=(7, 7))
    plt.hist(df, color='k', edgecolor='k')
    plt.xlim(-1, 1)
    plt.ylabel('Number of Statements', fontsize=30)
    plt.xticks([-1, 0, 1], fontsize=18, labels=['Negative', 'Neutral', 'Positive'])
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=24)
    plt.show()

#%% Hist plots
hist_plot(data['raw_compound'], 'Raw Statements') # All sentiment raw
hist_plot(data['cleaned_compound'], 'Cleaned Statements') # All senitment lemmatized

#%% Density plot with shade
import seaborn as sns
def density_plot(df, title):
    fig = plt.subplots(figsize=(7, 7))
    sns.kdeplot(df, shade='True', color='k', legend=False)
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.ylabel('Probability of Sentiment', fontsize=30)
    plt.xticks([-1, 0, 1], fontsize=18, labels=['Negative', 'Neutral', 'Positive'])
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=24)
    plt.show()

#%% Sentiment plots
density_plot(data['raw_compound'], title='Raw Input')
density_plot(data['cleaned_compound'], title='Cleaned Input')

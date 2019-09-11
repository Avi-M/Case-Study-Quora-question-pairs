
# coding: utf-8

# ### Description and  Problem Statement:
# 

# #### Quora is a social media platform where people ask questions and can connect to the actual experts who contribute unique insights and quality answers. Quora has an enormous user base and over 100 million people visit Quora every month,so it’s no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause readers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. so the PROBLEM is:
# #### 1.Identify which questions asked on Quora are duplicates of questions that have already been asked.
# #### 2.This could be useful to instantly provide answers to questions that have already been answered.
# #### 3.We are tasked with predicting whether a pair of questions are duplicates or not.
# 
#  

# ### DATA:
# <pre>
# The data is from Kaggle (Quora Question Pairs) and contains a human-labeled training set and a test set. Each record in the training set represents a pair of questions and a binary label indicating if it is a duplicate or not.
# The data, made available for non-commercial purposes (https://www.quora.com/about/tos) in a Kaggle competition(https://www.kaggle.com/c/quora-question-pairs) and on Quora’s blog (https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) consists of 404,290 question pairs having the following format:
#  • id - the id question pair
#  • qid1, qid2 - unique ids of each question
#  • question1, question2 - the full text of each question
#  • is duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.
# </pre> 
# 

# ### Performance Metric:
# ### Metric(s):
# <pre> a. log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
#  b. Binary Confusion Matrix </pre>

# ### DATA OVERVIEW:
# <pre>
# "id","qid1","qid2","question1","question2","is_duplicate"
# "0","1","2","What are the some good movies to watch?","What are the best movies to watch?","0"
# "1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
# "7","15","16","Do dentists earn more than other docter?","Do dentists earn more than other docter?why?","0"
# "11","23","24","Should i wait for iPad Air 3 or purchase the iPad Air 2 ?","should i buy the iPad Air or wait for next iPad Air ?","0"
# "9","45","46","How can I be a good geologist?","What should I do to be a great geologist?","1"
# "14","33","34","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
# </pre>
# 

# ### Train and Test Construction:
# <pre>We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.<pre>

# ### Exploratory Data Analysis:
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc
import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


# In[2]:


df = pd.read_csv("F:\\ML Pro\\train.csv")

print("Number of data points:",df.shape[0])


# In[3]:


df.head()


# In[4]:


df.info()


# ### Distribution of data points among output classes:
# 

# In[5]:


df.groupby("is_duplicate")['id'].count().plot.bar() #Number of duplicate(smilar) and non-duplicate(non similar) questions


# In[6]:


print('~> Total number of question pairs for training:\n   {}'.format(len(df)))


# In[7]:


print('~> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))
print('\n~> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(df['is_duplicate'].mean()*100, 2)))


# ### Number of unique questions:

# In[8]:


qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
unique_qs = len(np.unique(qids))
qs_morethan_onetime = np.sum(qids.value_counts() > 1)
print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))
#print len(np.unique(qids))
print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))
print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 
q_vals=qids.value_counts()
q_vals=q_vals.values


# In[9]:


x = ["unique_questions" , "Repeated Questions"]
y =  [unique_qs , qs_morethan_onetime]

plt.figure(figsize=(10, 6))
plt.title ("Plot representing unique and repeated questions  ")
sns.barplot(x,y)
plt.show()


# In[10]:


#checking whether there are any repeated pair of questions

pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()

print ("Number of duplicate questions:",(pair_duplicates).shape[0] - df.shape[0])


# ###  No. of occurrences of each questions:

# In[11]:


plt.figure(figsize=(20, 10))

plt.hist(qids.value_counts(), bins=160)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 


# In[12]:


#Checking whether there are any rows with null values
nan_rows = df[df.isnull().any(1)]
print (nan_rows)


# In[13]:


df = df.fillna('')
nan_rows = df[df.isnull().any(1)]
print (nan_rows)


# ###  Basic Feature Extraction (before cleaning):  

# Let us now construct a few features like:
#  - ____freq_qid1____ = Frequency of qid1's
#  - ____freq_qid2____ = Frequency of qid2's 
#  - ____q1len____ = Length of q1
#  - ____q2len____ = Length of q2
#  - ____q1_n_words____ = Number of words in Question 1
#  - ____q2_n_words____ = Number of words in Question 2
#  - ____word_Common____ = (Number of common unique words in Question 1 and Question 2)
#  - ____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
#  - ____word_share____ = (word_common)/(word_Total)
#  - ____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2 
#  - ____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2  

# In[14]:


if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')
else:
    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 
    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')
    df['q1len'] = df['question1'].str.len() 
    df['q2len'] = df['question2'].str.len()
    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)
    df['word_Common'] = df.apply(normalized_word_Common, axis=1)

    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * (len(w1) + len(w2))
    df['word_Total'] = df.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    df.to_csv("df_fe_without_preprocessing_train.csv", index=False)

df.head()


# ### Analysis of some of the extracted features:
# Here are some questions have only one single words.

# In[15]:


print ("Minimum length of the questions in question1 : " , min(df['q1_n_words']))

print ("Minimum length of the questions in question2 : " , min(df['q2_n_words']))

print ("Number of Questions with minimum length [question1] :", df[df['q1_n_words']== 1].shape[0])
print ("Number of Questions with minimum length [question2] :", df[df['q2_n_words']== 1].shape[0])


# ### Feature: word_share -

# In[16]:


plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )
plt.show()


# <ref>1. The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
# 2.The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar) <ref>

# ### Feature: word_common -

# In[17]:


plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue' )
plt.show()


# <ref> The distributions of the word_Common feature in similar and non-similar questions are highly overlapping </ref>

# ### EDA: Advanced Feature Extraction:
# 

# In[18]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
# This package is used for finding longest common subsequence between two strings
# you can write your own dp code for this
import distance
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
# Import the Required lib packages for WORD-Cloud generation
# https://stackoverflow.com/questions/45625434/how-to-install-wordcloud-in-python3-6
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image


# In[19]:


#https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')
    df = df.fillna('')
    df.head()
else:
    print("get df_fe_without_preprocessing_train.csv from drive or run the previous notebook")


# In[20]:


df.head(2)


# ### Preprocessing of Text :

# #### - Preprocessing:
#     - Removing html tags 
#     - Removing Punctuations
#     - Performing stemming
#     - Removing Stopwords
#     - Expanding contractions etc.

# In[21]:


# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
               
    
    return x
    


# ### <h2>  Advanced Feature Extraction (NLP and Fuzzy Features): </h2>

# ### Definition:
# 
# #### Token: You get a token by splitting sentence a space
# #### Stop_Word : stop words as per NLTK.
# #### Word : A token that is not a stop_word </ref>
# #### Features:
# #### cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
# #### cwc_min = common_word_count / (min(len(q1_words), len(q2_words))<ref>
# 
# #### cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
# #### cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
# #### csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
# #### csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
# #### csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
# #### csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
# #### ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
# #### ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
# #### ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
# #### ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
# #### last_word_eq : Check if First word of both questions is equal or not
# #### last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
# #### first_word_eq : Check if First word of both questions is equal or not
# #### first_word_eq = int(q1_tokens[0] == q2_tokens[0])
# #### abs_len_diff : Abs. length difference
# #### abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
# #### mean_len : Average Token Length of both Questions
# #### mean_len = (len(q1_tokens) + len(q2_tokens))/2
# #### fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# #### fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# #### token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# #### token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# #### longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2
# #### longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))</ref>

# In[22]:


def get_token_features(q1, q2):
    token_features = [0.0]*10
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df):
    # preprocessing each question
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    print("token features...")
    
    # Merging Features with dataset
    
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))
   
    #Computing Fuzzy Features and Merging with Dataset
    
    # do read this blog: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    # https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
    # https://github.com/seatgeek/fuzzywuzzy
    print("fuzzy features..")

    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df


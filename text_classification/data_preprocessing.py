
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm


# In[2]:


raw_data=pd.read_csv('train.csv',delimiter=',')
train={}
train_data=raw_data.values
label_one=np.zeros(6)
label_multi_count=Counter()
for sample in tqdm(train_data):
    id=sample[0]
    context=sample[1]
    labels=sample[2:]
    label_multi_count.update([str(labels)])
    for i,label in enumerate(labels):
        label_one[i]+=label
    train[id]={'context':context,'label':labels}


# In[3]:


label_num_ratio=[label_num/len(train_data) for label_num in label_one]
sum(label_multi_count.values())
label_num_ratio
label_multi_count


# In[3]:


raw_data=pd.read_csv('test.csv',delimiter=',')
test_data=raw_data.values
test={}
label_test_one=np.zeros(6)
label_multi_test_count=Counter()
for sample in tqdm(test_data):
    id=sample[0]
    context=sample[1]
    labels=sample[2:]
    label_multi_test_count.update([str(labels)])
    for i,label in enumerate(labels):
        label_one[i]+=label
    test[id]={'context':context}


# In[47]:


label_num_test_ratio=[label_num/len(test_data) for label_num in label_test_one]
sum(label_multi_test_count.values())
label_num_test_ratio
label_multi_test_count


# In[49]:


test_data[0]


# In[50]:


import json


# In[76]:


with open('train.json','w') as f_train, open('test.json','w') as f_test:
    json.dump(train,f_train)
    json.dump(test,f_test)


# In[4]:


len(train)


# In[5]:


len(test)


# In[4]:


#数据清洗
import nltk
import re
import nltk
import ssl
from nltk.corpus import stopwords
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
print('loading complete')


# In[5]:


from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
import string


# In[6]:


text=train['014bb932bd289352']['context'] 
text


# In[15]:


#文本预处理思路
"""
1,转换大小写
2，分句
3，分词，词型还原
4，去停用词
"""
import string 
from nltk.corpus import wordnet as wn  
from nltk.corpus import stopwords
def SenToken(document):#分割成句子  
    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')  
    sents =sent_tokenizer.tokenize(document)  
    return sents

def CleanLines(line):
    identify= line.maketrans(
       # If you find any of these
       "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&()*+,-./:;<=>?@[]^_`{|}~'\\",
       # Replace them by these
       "abcdefghijklmnopqrstuvwxyz                                          ")
    #delEStr = string.punctuation+string.digits  #ASCII 标点符号，数字
    cleanLine = line.translate(identify) #去掉ASCII 标点符号和空格
    return cleanLine  

def WordTokener(sent):#将单句字符串分割成词   
    wordsInStr= nltk.word_tokenize(sent)  
    return wordsInStr  

def CleanWords(word_list):#去掉标点符号，长度小于3的词以及non-alpha词，小写化  
    cleanWords=[] 
    cleanWords = [w for w in word_list if w not in stopwords.words('english') and 3<=len(w)]
    return cleanWords 

def StemWords(cleanWordsList):  
    stemWords=[]  
#   porter = nltk.PorterStemmer()#有博士说这个词干化工具效果不好，不是很专业  
#   result=[porter.stem(t) for t incleanTokens]   
    stemWords=[wn.morphy(w) for w in cleanWordsList]  
    return stemWords


# In[16]:


def preprocess_document(text):
    sents=SenToken(text)
    sents_list=[]
    for sent in sents:
        clear_sent=CleanLines(sent)
        word_list=WordTokener(clear_sent)
        clear_word_list=CleanWords(word_list)
        #stem_word_list=StemWords(clear_word_list)
        word_str=' '.join(clear_word_list)
        sents_list.append(word_str)
    sents_str='#'.join(sents_list)
    return sents_str


# In[17]:


pre_text=preprocess_document(text)
pre_text


# In[18]:


pre_train={}
i=0
for id,sample in tqdm(train.items()):
    i+=1
    context=sample['context']
    pre_context=preprocess_document(context)
    pre_train[id]={'pre_context':pre_context,'label':sample['label']}


# In[13]:





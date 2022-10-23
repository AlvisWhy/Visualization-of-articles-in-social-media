#!/usr/bin/env python
# coding: utf-8

#25]:


# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections  
import itertools
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from wordcloud import WordCloud,ImageColorGenerator,random_color_func #词云，颜色生成器，停止词
from PIL import Image #处理图片
import warnings
warnings.filterwarnings("ignore")


#29]:

#====预处理函数
replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'cannot'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in
        patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s
replace = RegexpReplacer()
#删除网址
def _remove_url(line):
    line = re.sub("http[s]?:\S+|ftp:\S+|www.\S+",' ', line)
    return line
#只保留英文字符，数字，和这些符号
def remove_punctuation(line):
    line = re.sub("[^ ^a-z^A-Z^0-9]",' ', line)
    return line
#预处理
def deal_line(line,stop_word=False):
    line = _remove_url(line)
    line = remove_punctuation(line)
    line = line.lower() #全部转为小写
    #分词
    line = word_tokenize(line)
    #词形还原（归并）
    wnl = WordNetLemmatizer()
    line = [wnl.lemmatize(ws) for ws in line]
    #去除停用词
    if stop_word:
        line = [w for w in line if not w in stopwords and len(w) > 1]
    return line
stopwords = nltk_stopwords.words('english')


#10]:

#===模型预测函数
labels = {0:'sadness',
1:'joy',
2:'love',
3:'anger',
4:'fear',
5:'surprise'}
def predict_sentiment(text):
    text_cut = text.astype(str).map(lambda x:deal_line(x))
    text_tokens = text_cut.map(lambda x:[wvmodel.wv.key_to_index[i] for i in x if i in wvmodel.wv.key_to_index.keys()])#编码
    #序列化，填充一致
    X = pad_sequences(text_tokens.values,
                      maxlen=maxlen,
                      padding='pre', 
                      truncating='pre')
    X[X>=max_word] = 0 #只保留最大单词数
    y_pred = pd.DataFrame(model_cnn.predict(X),columns=labels.values())
    return y_pred


#4]:

#===读取word2vec模型
wvmodel = Word2Vec.load("wvmodel.model")
max_word =  len(wvmodel.wv)#最大单词数
maxlen = 40 #文本序列化的数量


#5]:
"""
#读取cnn模型
model_cnn = load_model(
    'emotion_model_CNN.h5')


#6]:

#读取数据
data = pd.read_csv('retweet.csv')

#data = pd.read_csv('t_comment.csv')

#17]:

#模型预测
senti_pred = predict_sentiment(data['source_content'])#预测情感值
#最大的情感是什么
senti_pred['source_emotion'] = pd.Series(np.argmax(senti_pred.iloc[:,:6].values,axis=1)).map(lambda x:labels[x])
#最大的情感的概率
senti_pred['max_source_emotion'] = senti_pred.apply(lambda x:x[x['source_emotion']],axis=1)
data = pd.concat([data,senti_pred],axis=1) #数据拼接


#22]:






# # 词频词云

#30]:


mystopwords = [line.rstrip().lower() for line in open(
    '../../Documents/WeChat Files/wxid_ej3rwim596kq22/FileStorage/File/2022-07/7-19/mystopwords.txt', encoding='utf-8')]
stopwords = nltk_stopwords.words('english')+mystopwords+[''+' ']

print(len(data))
dell = []
for i in range(0,data.shape[0]):
    if data.loc[i,'max_emotion']== "joy" or data.loc[i,'max_emotion']=="love":
        dell.append(i)
print(len(dell))
data = data.drop(dell)

text_cut = data['content'].astype(str).map(lambda x:deal_line(x,stop_word=True))


#31]:

all_words = list(itertools.chain(*text_cut)) #全部的单词
word_counts = collections.Counter(all_words)  #做词频统计

print(word_counts)
word_counts_top = word_counts.most_common()# 获取前N最高频的词####-------------重要的
pd.DataFrame(word_counts_top,columns=['word','count']).to_excel('wordCountNeg.xlsx',index=0) #保存词频统计结果
"""
#32]:


#===词云图
"""
data = pd.read_excel("wordCount.xlsx")
word_counts = collections.Counter( dict(zip(data['word'],data['count'])))
backgroud = np.array(Image.open(
    '111.png'))  #词云底图


"""
data = pd.read_excel("wordCountNeg.xlsx")
word_counts = collections.Counter( dict(zip(data['word'],data['count'])))
backgroud = np.array(Image.open(
    '222.png'))  #词云底图

"""
data = pd.read_excel("wordCountPos.xlsx")

word_counts = collections.Counter( dict(zip(data['word'],data['count'])))
backgroud = np.array(Image.open(
    '333.png'))  #词云底图
"""

wc = WordCloud(width=3240, height=2160,
        background_color='black',
        mode='RGB', 
        mask=backgroud, #添加蒙版，生成指定形状的词云，并且词云图的颜色可从蒙版里提取
        max_words=400,
        font_path='C:\Windows\Fonts\STZHONGS.ttf',
        max_font_size=200,
        relative_scaling=0.5, #设置字体大小与词频的关联程度为0.6
        random_state=100,
        scale=2 ,
        colormap=sns.color_palette("husl", 10,as_cmap=True),
        margin=1
        ).fit_words(word_counts)

image_color = ImageColorGenerator(backgroud)#设置生成词云的颜色，如去掉这两行则字体为默认颜色
wc.recolor(color_func=image_color)

f, ax = plt.subplots(figsize=(10,9))  #画布大小
plt.imshow(wc,interpolation="bilinear") #显示词云
plt.axis('off') #关闭x,y轴
plt.show()#显示
wc.to_file('词云图.jpg') #保存词云图



#42]:
"""
#获取评论的主题
def get_topic(x):
    x = re.findall(r'(#)(.*?)(\w+)', x)
    x = ';'.join([''.join(i[-1]) for i in x])
    return x


#45]:


data['topic'] = data['content'].map(lambda x:get_topic(x))
"""
#46]:
"""
#保存数据
data.to_csv('retweet+emotion_topic.csv',index=0)

#data.to_csv('t_comment+emotion_topic.csv',index=0)

# ]:
"""
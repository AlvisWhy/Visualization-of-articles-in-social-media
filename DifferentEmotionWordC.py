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

data = pd.read_csv('t_article+emotion_topic.csv')

dataPos= pd.DataFrame()
dataNeg= pd.DataFrame()
dataNeu= pd.DataFrame()

for i in range(data.shape[0]):
    if data.loc[i,'max_emotion'] == 'sadness' or data.loc[i,'max_emotion'] == 'anger' or data.loc[i,'max_emotion'] == 'fear':
        dataNeg.append(data.loc[i])
    elif data.loc[i,'max_emotion'] == 'love' or data.loc[i,'max_emotion'] == 'joy':
        dataPos.append(data.loc[i])
    else:
        dataNeu.append(data.loc[i])




def generateCloud(data):
    mystopwords = [line.rstrip().lower() for line in open(
        '../../Documents/WeChat Files/wxid_ej3rwim596kq22/FileStorage/File/2022-07/7-19/mystopwords.txt', encoding='utf-8')]
    stopwords = nltk_stopwords.words('english')+mystopwords+[''+' ']
    text_cut = data['content'].astype(str).map(lambda x:deal_line(x,stop_word=True))


    #31]:


    all_words = list(itertools.chain(*text_cut)) #全部的单词
    word_counts = collections.Counter(all_words)  #做词频统计
    word_counts_top = word_counts.most_common()# 获取前N最高频的词####-------------重要的
    pd.DataFrame(word_counts_top,columns=['word','count']).to_excel('wordCount.xlsx',index=0) #保存词频统计结果


    backgroud = np.array(Image.open(
        '../../Documents/WeChat Files/wxid_ej3rwim596kq22/FileStorage/File/2022-07/7-19/大树底图.jpg'))  #词云底图
    wc = WordCloud(width=3000, height=2200,
            background_color='white',
            mode='RGB',
            mask=backgroud, #添加蒙版，生成指定形状的词云，并且词云图的颜色可从蒙版里提取
            max_words=100,
            font_path='C:\Windows\Fonts\STZHONGS.ttf',
            max_font_size=150,
            relative_scaling=0.6, #设置字体大小与词频的关联程度为0.6
            random_state=50,
            scale=2 ,
            colormap=sns.color_palette("husl", 10,as_cmap=True)
            ).fit_words(word_counts)

    image_color = ImageColorGenerator(backgroud)#设置生成词云的颜色，如去掉这两行则字体为默认颜色
    wc.recolor(color_func=image_color)

    f, ax = plt.subplots(figsize=(10,9))  #画布大小
    plt.imshow(wc,interpolation="bilinear") #显示词云
    plt.axis('off') #关闭x,y轴
    plt.show()#显示
    wc.to_file('词云图.jpg') #保存词云图

generateCloud(dataPos)
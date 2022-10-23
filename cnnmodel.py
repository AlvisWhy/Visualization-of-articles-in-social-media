#!/usr/bin/env python
# coding: utf-8

#1]:


# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections  
import itertools
plt.rcParams['font.sans-serif'] = ['FangSong'] # 中文字体设置-黑体SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import warnings
warnings.filterwarnings("ignore")


#2]:

#===读取数据
train = pd.read_csv('training.csv')
val = pd.read_csv('validation.csv')
test = pd.read_csv('test.csv')


#5]:


print(train['label'].value_counts())


#=========预处理


from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords



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
def deal_line(line):
    line = _remove_url(line)
    line = remove_punctuation(line)
    line = line.lower() #全部转为小写
    #分词
    line = word_tokenize(line)
    #词形还原（归并）
    wnl = WordNetLemmatizer()
    line = [wnl.lemmatize(ws) for ws in line]
    #去除停用词
    #line = [w for w in line if not w in stopwords and len(w) > 1]
    return line
stopwords = nltk_stopwords.words('english')

#分词
train_cut = train['text'].astype(str).map(lambda x:deal_line(x))
val_cut = val['text'].astype(str).map(lambda x:deal_line(x))
test_cut = test['text'].astype(str).map(lambda x:deal_line(x))


#9]:

#全部数据合并
text_cut = pd.concat([train_cut,val_cut,test_cut],ignore_index=True)




#词长度
text_cut.map(lambda x:len(x)).hist(bins=30)
plt.show()
plt.savefig('text-cnn_len.png', dpi=600, bbox_inches='tight')
#11]:


text_len = text_cut.map(lambda x:len(x)).value_counts('0').sort_index().cumsum()
plt.plot(text_len)
plt.show()
plt.savefig('text-cnn_len2.png', dpi=600, bbox_inches='tight')

#==========WORD2VEC

#12]:


from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#https://radimrehurek.com/gensim/models/word2vec.html 模型参数说明


#13]:


vector_size=300 #word2vec的维度
maxlen = 40 #文本序列化的数量
#训练word2vec模型
wvmodel = Word2Vec(sentences=text_cut,vector_size=vector_size,window=10,min_count=5)
max_word = len(wvmodel.wv) #最大单词数
print('word2vec单词数量：',len(wvmodel.wv))
wvmodel.save("wvmodel.model") #保存模型

#获得每一个词的词向量
embedding_matrix = np.zeros((max_word, vector_size))
for i in range(max_word):
    embedding_matrix[i, :] = wvmodel.wv[wvmodel.wv.index_to_key[i]]

#===将文本标记化，用来输入模型===
def get_X(text_cut):
    # 标记化
    text_tokens = text_cut.map(lambda x:[wvmodel.wv.key_to_index[i] for i in x if i in wvmodel.wv.key_to_index.keys()])#编码
    #序列化，填充一致
    X = pad_sequences(text_tokens.values,
                      maxlen=maxlen,
                      padding='pre',  #在前面填充
                      truncating='pre')
    X[X>=max_word] = 0 #只保留最大单词数
    return X


#18]:


X_train = get_X(train_cut)
X_val = get_X(val_cut)
X_test = get_X(test_cut)
y_train = train['label']
y_val = val['label']
y_test = test['label']
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


#=====CNN=====

import tensorflow as tf
from tensorflow.keras.callbacks import Callback,EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Embedding
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D,concatenate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,auc,RocCurveDisplay
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


#20]:


#创建训练测试集验证数据
batch_size = 128
ds_train_encoded = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(10000).batch(batch_size)
ds_val_encoded = tf.data.Dataset.from_tensor_slices((X_val,y_val)).batch(batch_size)
ds_test_encoded = tf.data.Dataset.from_tensor_slices((X_test,y_test)).batch(batch_size)


#21]:


#========搭建模型===========
def bulid_cnn_model():
    input_ = Input(shape=(maxlen,))
    embedding = Embedding(max_word,vector_size,weights=[embedding_matrix],trainable=True)(input_)
    c1 = Conv1D(198,3,activation='relu',strides=1)(embedding)
    m1 = GlobalMaxPooling1D()(c1)
    c2 = Conv1D(198,4,activation='relu',strides=1)(embedding)
    m2 = GlobalMaxPooling1D()(c2)
    c3 = Conv1D(198,5,activation='relu',strides=1)(embedding)
    m3 = GlobalMaxPooling1D()(c3)
    con = concatenate([m1,m2,m3],axis=-1)
    dense0 = Dropout(0.5)(con)
    dense1 = Dense(6,activation='softmax')(dense0)
    model = Model(inputs=input_, outputs=dense1)
    model.summary()
    return model


#22]:


model_cnn = bulid_cnn_model()


#23]:


#设置模型
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model_cnn.compile(optimizer='adam', loss=loss, metrics=[metric])
earlystopping = EarlyStopping(monitor='val_accuracy', patience=5)  # 若5个epoch没有提高则early_stopping


#24]:


#训练模型
history_cnn = model_cnn.fit(ds_train_encoded , epochs=50,validation_data=ds_val_encoded,callbacks=[earlystopping])


#41]:


pd.DataFrame(history_cnn.history).to_excel('cnn模型过程.xlsx')


#25]:


#====绘制曲线
plt.plot(history_cnn.history['accuracy'],label='train')
plt.plot(history_cnn.history['val_accuracy'],label='val')
plt.title('Accuracy')
plt.legend(loc=4)
plt.savefig('cnn_Accuracy.png', dpi=600, bbox_inches='tight')
plt.show()
plt.plot(history_cnn.history['loss'],label='train')
plt.plot(history_cnn.history['val_loss'],label='val')
plt.title('Loss')
plt.legend(loc=1)
plt.savefig('cnn_Loss.png', dpi=600, bbox_inches='tight')
plt.show()



labels = {0:'sadness',
1:'joy',
2:'love',
3:'anger',
4:'fear',
5:'surprise'}



y_test_pred_proba_cnn = model_cnn.predict(ds_test_encoded)
y_test_pred_cnn = np.argmax(y_test_pred_proba_cnn,axis=1)


#34]:


print(classification_report(y_test, y_test_pred_cnn,digits=4,target_names=labels.values()))


#35]:


f, ax = plt.subplots(figsize=(8,7))  #画布大小
cm = confusion_matrix(y_test, y_test_pred_cnn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=labels.values())
disp.plot(values_format='d', cmap=plt.cm.Blues,ax=ax)
plt.xticks(rotation=60) #x轴不同位置的名称
plt.savefig('text-cnn_test_cm.png', dpi=600, bbox_inches='tight')
plt.show()


#37]:


model_cnn.save('emotion_model_CNN.h5')


# # 词向量降维

#55]:


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap


#56]:


word_embedding = pd.DataFrame(embedding_matrix,index=wvmodel.wv.key_to_index.keys())


#57]:

#==pca降维
pca = PCA(n_components=2)
pd.DataFrame(pca.fit_transform(word_embedding),index=word_embedding.index).to_csv(
    'P2D')
pca = PCA(n_components=3)
pd.DataFrame(pca.fit_transform(word_embedding),index=word_embedding.index).to_csv(
    'P3D')


#58]:

#isomap 降维
isomap = Isomap(n_components=2)
pd.DataFrame(isomap.fit_transform(word_embedding),index=word_embedding.index).to_csv(
    'I2D')
isomap = Isomap(n_components=3)
pd.DataFrame(isomap.fit_transform(word_embedding),index=word_embedding.index).to_csv('I3D')


# ]:





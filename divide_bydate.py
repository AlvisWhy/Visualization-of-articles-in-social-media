import numpy as np
import pandas as pd


df_article = pd.read_csv("t_comment+emotion_topic.csv")

"""
#情感阈值分类
for i in range(df_article.shape[0]):
    if df_article.loc[i,'max_emotion_proba'] < 0.4:
        df_article.loc[i, 'max_emotion'] = 'neutural'

df_article.to_csv("t_comment+emotion_topic.csv")
print("over")

emotion_statistics = pd.DataFrame(columns=['date','count','sadness','joy','love','anger','fear','surprise','neutural'])
"""

#分日期输出
def Bydate(pubdate,emotion):
    s = 0
    e0 = 0
    e1 = 0
    e2 = 0
    e3 = 0
    e4 = 0
    e5 = 0
    e6 = 0

    newdf = df_article.drop(index=df_article.index)
    for i in range(df_article.shape[0]):
        if df_article.loc[i,'cmt_time'][0:10] == pubdate:
            newdf = newdf.append(df_article.loc[i])
            s += 1
            if df_article.loc[i,'max_emotion'] == 'sadness':
                e0 += 1
            elif df_article.loc[i,'max_emotion'] == 'joy':
                e1 += 1
            elif df_article.loc[i,'max_emotion'] == 'love':
                e2 += 1
            elif df_article.loc[i,'max_emotion'] == 'anger':
                e3+= 1
            elif df_article.loc[i,'max_emotion'] == 'fear':
                e4 += 1
            elif df_article.loc[i,'max_emotion'] == 'surprise':
                e5 += 1
            elif df_article.loc[i,'max_emotion'] == 'neutural':
                e6 += 1
    es0 = e0 /s
    es1 = e1 / s
    es2 = e2 / s
    es3 = e3 / s
    es4 = e4 / s
    es5 = e5 / s
    es6 = e6 / s

    newdf.to_csv(pubdate + " comment.csv")
    emotion = emotion.append({'date':pubdate,'count':s,'sadness':es0 ,'joy':es1,'love':es2,'anger':es3,'fear':es4,'surprise':es5,'neutural':es6},ignore_index=True)
    return emotion


emotion_statistics = Bydate("2022-06-25",emotion_statistics)
emotion_statistics = Bydate("2022-06-26",emotion_statistics)
emotion_statistics = Bydate("2022-06-27",emotion_statistics)
emotion_statistics = Bydate("2022-06-28",emotion_statistics)
emotion_statistics = Bydate("2022-06-29",emotion_statistics)
emotion_statistics = Bydate("2022-06-30",emotion_statistics)
emotion_statistics = Bydate("2022-07-01",emotion_statistics)
emotion_statistics = Bydate("2022-07-02",emotion_statistics)
emotion_statistics = Bydate("2022-07-03",emotion_statistics)
emotion_statistics = Bydate("2022-07-04",emotion_statistics)
emotion_statistics = Bydate("2022-07-05",emotion_statistics)
emotion_statistics = Bydate("2022-07-06",emotion_statistics)
emotion_statistics = Bydate("2022-07-07",emotion_statistics)
emotion_statistics = Bydate("2022-07-08",emotion_statistics)
emotion_statistics = Bydate("2022-07-09",emotion_statistics)
emotion_statistics = Bydate("2022-07-10",emotion_statistics)
emotion_statistics = Bydate("2022-07-11",emotion_statistics)
emotion_statistics = Bydate("2022-07-12",emotion_statistics)
emotion_statistics = Bydate("2022-07-13",emotion_statistics)
emotion_statistics = Bydate("2022-07-14",emotion_statistics)
emotion_statistics = Bydate("2022-07-15",emotion_statistics)
emotion_statistics = Bydate("2022-07-16",emotion_statistics)
emotion_statistics = Bydate("2022-07-17",emotion_statistics)
emotion_statistics = Bydate("2022-07-18",emotion_statistics)
emotion_statistics = Bydate("2022-07-19",emotion_statistics)
emotion_statistics = Bydate("2022-07-20",emotion_statistics)
emotion_statistics = Bydate("2022-07-21",emotion_statistics)
emotion_statistics = Bydate("2022-07-22",emotion_statistics)

emotion_statistics.to_csv('emotion_statistics_c.csv')


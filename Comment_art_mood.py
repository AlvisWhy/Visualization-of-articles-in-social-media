import numpy as np
import pandas as pd
df_comment = pd.read_csv("2022-06-28 comment.csv")
df_article = pd.read_csv("t_article+emotion.csv")


k = 0
s = 0

for i in range(df_comment.shape[0]):
        for j in range(0,df_article.shape[0]):
            if  df_comment.loc[i,'tweet_id'] == df_article.loc[j,'article_id']:
                df_comment.loc[i,'article_emotion'] = df_article.loc[j,'max_emotion']
                s = j
                break
        k += 1
        if k > 20000:
            break
        print(k)

df_comment.to_csv("comment_after_pair_6-28.csv")
print(df_comment.head())
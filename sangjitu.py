import pandas as pd
import numpy as np

data = pd.read_csv("comment_after_pair_6-28.csv")

sangjitu = pd.DataFrame(columns=['sadness','joy','love','anger','fear','surprise','neutural'],index= ['sadness','joy','love','anger','fear','surprise','neutural'])
sangjitu = sangjitu.fillna(value=0)
print(sangjitu)

for i in range(len(data)):
    if (data.loc[i,'max_emotion'] == data.loc[i,'max_emotion']) and (data.loc[i,'article_emotion'] == data.loc[i,'article_emotion']) :

        emo = data.loc[i,'max_emotion']
        emoS = data.loc[i,'article_emotion']
        print(emoS)
        sangjitu.loc[emoS,emo] += 1

print(sangjitu)
sangjitu.to_csv('sangjitu3.csv')
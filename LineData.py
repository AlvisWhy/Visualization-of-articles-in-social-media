
import pandas as pd

lista = []
listtb = []
k1 = 0
k2 = 0
k3 = 0
k4 = 0
k5 = 0
k6 = 0

M1 = 0
M2 = 0
M3 = 0

S1 = 0
S2 = 0
S3 = 0


data_dir = 'retweet+emotion.csv'
sDf = pd.read_csv(data_dir, encoding='ISO-8859-1')
df = pd.DataFrame(columns=['Source','Target','Weight'])
dfd = pd.DataFrame(columns=['ID','Label','modularity_class','cluster'])

emo_plo = {"p-p":1, "p-ng":2, "p-nl":3, "nl-nl":4,"nl-ng":5,"ng-ng":6, "ng-p":2, "nl-p":3, "ng-nl":5}

for i in range(len(sDf)):

    if sDf.loc[i,"max_emotion"] =='sadness' or sDf.loc[i,"max_emotion"] =='anger' or sDf.loc[i,"max_emotion"] =='fear':
        aa = 'ng'
        M1 += 1
    elif sDf.loc[i,"max_emotion"] =='joy' or sDf.loc[i,"max_emotion"] =='love' :
        aa = "p"
        M2 += 1
    elif sDf.loc[i,"max_emotion"] =='surprise' or sDf.loc[i,"max_emotion"] =='neutural' :
        aa = 'nl'
        M3 += 1

    if sDf.loc[i, "source_emotion"] =='sadness' or sDf.loc[i, "source_emotion"] =='anger' or sDf.loc[i, "source_emotion"] =='fear':
        bb = 'ng'
    elif sDf.loc[i, "source_emotion"]=='joy' or sDf.loc[i, "source_emotion"]=='love' :
        bb = "p"
    elif sDf.loc[i, "source_emotion"]=='surprise' or sDf.loc[i, "source_emotion"] =='neutural' :
        bb = 'nl'

    t = emo_plo[aa + '-' + bb]

    c = {'Source':"k_"+str(sDf.loc[i,"user_id"]),'Target':"k_" +str(sDf.loc[i,"source_article_id"]),'Weight':t}
    a = {'ID':"k_"+str(sDf.loc[i,"user_id"]),"Label":"k_"+str(sDf.loc[i,"user_id"]),"modularity_class":1,"cluster": sDf.loc[i,"max_emotion"]}
    b = {'ID':"k_"+ str(sDf.loc[i, "source_article_id"]),"Label":"k_"+ str(sDf.loc[i, "source_article_id"]), "modularity_class":1, "cluster": sDf.loc[i, "source_emotion"]}
    df = df.append(c,ignore_index= True)
    dfd = dfd.append(a,ignore_index=True)
    dfd = dfd.append(b,ignore_index=True)

for i in range(0,df.shape[0]):
    if df.loc[i,"Weight"] == 1:
        k1+=1

    elif df.loc[i,"Weight"] == 2:
        k2+=1

    elif df.loc[i,"Weight"] == 3:
        k3+=1

    elif df.loc[i,"Weight"] == 4:
        k4+=1

    elif df.loc[i,"Weight"] == 5:
        k5+=1

    elif df.loc[i,"Weight"] == 6:
        k6+=1

sum = M1 + M2 +M3
print(M1/sum,M2/sum,M3/sum)
print()
import pandas as pd
import re

df_parquet=pd.read_parquet('C:/Users/linkm/Documents/GitHub/IA/data/raw/fineweb.parquet', columns=['text'])

df_train=df_parquet.truncate(before=1,after=10000,axis=0)

list_train=df_train['text'].tolist()
processed_dataset=[]

for article in list_train:
    article=article.replace(" ", "_")
    article_list=re.findall(r'\_?[a-zA-Zà-ÿ]+|_?\d+|_?[^\w\s_]+' , article)

with open('C:/Users/linkm/Documents/GitHub/IA/data/raw/articles','w'):
    

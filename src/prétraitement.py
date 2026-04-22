import pandas as pd
import re

pattern = re.compile(r'▁?[a-zA-Zà-ÿ]+|▁?\d+|▁?[^\w\s ]+')

df_parquet=pd.read_parquet('C:/Users/linkm/Documents/GitHub/IA/data/raw/fineweb.parquet', columns=['text'])

with open('C:/Users/linkm/Documents/GitHub/IA/src/fineweb.txt','w') as f:

    for i, article in enumerate(df_parquet['text']):
        
        article = article.replace(" ", "▁")
        article_list = pattern.findall(article)

        buffer_article = []

        for word in article_list:
            for byte in word.encode(): 
                buffer_article.append(str(byte))
                
        buffer_article.append("-1")
























df_train=df_parquet
list_train=df_train['text'].tolist()
processed_dataset=[]

for article in list_train:
    
    article = article.replace(" ", "▁")
    article_list = re.findall(r'▁?[a-zA-Zà-ÿ]+|▁?\d+|▁?[^\w\s ]+', article)

    article_bytes=[bytearray(word.encode()) for word in article_list]
    processed_dataset.append(article_bytes)



    for article in processed_dataset :

        buffer_article=[]

        for word in article :
            for x in word :
                buffer_article.append(str(x))
        buffer_article.append("-1")

        f.write(" ".join(buffer_article) + " ")
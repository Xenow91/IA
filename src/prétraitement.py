import pandas as pd
import re

pattern = re.compile(r'▁?[a-zA-Zà-ÿ]+|▁?\d+|▁?[^\w\s ]+')

df_parquet = pd.read_parquet('C:/Users/linkm/Documents/GitHub/IA/data/raw/fineweb.parquet', columns=['text'])

with open('C:/Users/linkm/Documents/GitHub/IA/src/fineweb.txt', 'w') as f:
    
    for i, article in enumerate(df_parquet['text']):
        
        article = article.replace(" ", "▁")
        article_list = pattern.findall(article)

        buffer_article = []
        
        for word in article_list:
            for byte in word.encode(): 
                buffer_article.append(str(byte))
                
        buffer_article.append("-1")

        f.write(" ".join(buffer_article) + " ")

        if (i + 1) % 10000 == 0:
            print(f"Traités : {i + 1} / {len(df_parquet)}")
import pandas as pd
import re
import pickle
from collections import Counter

df_parquet=pd.read_parquet('C:/Users/linkm/Documents/GitHub/IA/data/raw/fineweb.parquet', columns=['text'])

df_train=df_parquet.truncate(before=1,after=10000,axis=0)

list_train=df_train['text'].tolist()
processed_dataset=[]

for article in list_train:
    article=article.replace(" ", "_")
    article_list=re.findall(r'\_?[a-zA-Zà-ÿ]+|_?\d+|_?[^\w\s_]+' , article)

    article_bytes=[bytearray(word.encode()) for word in article_list]
    processed_dataset.append(article_bytes)

word_freqs=Counter()
for article in processed_dataset:
    for word in article:
        word_freqs[tuple(word)]+=1


vocab={(i,):i for i in range(256)}
inverse_vocab=[(i,) for i in range(256)]


def count():
    pairs=Counter()
    for word in word_freqs:
        for a,b in zip(word,word[1:]):
            pairs[a,b]+=word_freqs[word]
    return pairs


def merge(id1, id2, new_id):
    new_word_freqs=Counter()
    for word in word_freqs:
        if id1 not in word:
            new_word_freqs[word] = word_freqs[word]
            continue

        k=0
        new_word=[]
        while k<len(word):
            if word[k]!=id1 :
                new_word.append(word[k])
                k+=1
                continue
            if (k+1)<len(word) and word[k+1]==id2 :
                new_word.append(new_id)
                k+=2
                continue
            new_word.append(word[k])
            k+=1

        new_word=tuple(new_word)
        
        new_word_freqs[new_word]=word_freqs[word]
    
    return new_word_freqs


taille=256
while taille<100 :
    pairs=count()

    max=pairs.most_common(1)
    byte1,byte2=max[0][0]

    vocab[(byte1,byte2)]=taille
    inverse_vocab.append((byte1,byte2))

    word_freqs=merge(byte1,byte2,taille)

    taille+=1

with open ('C:/Users/linkm/Documents/GitHub/IA/data/processed/vocab.pkl', 'wb') as f:
    pickle.dump(vocab,f)
with open ('C:/Users/linkm/Documents/GitHub/IA/data/processed/inverse_vocab.pkl', 'wb') as f:
    pickle.dump(inverse_vocab,f)
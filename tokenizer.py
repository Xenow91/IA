import re

class Tokenizer:
    def __init__(self, vocab_path: str, merges_path: str):

        self.vocab = {} 
        self.merges = {}
        self.pattern = re.compile(r'▁?[a-zA-Zà-ÿ]+|▁?\d+|▁?[^\w\s ]+')
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    tok_id = int(parts[0])
                    byte_array = bytes([int(b) for b in parts[1].split(',') if b])
                    self.vocab[tok_id] = byte_array

        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                left, right, new_id = map(int, line.strip().split())
                self.merges[(left, right)] = new_id

    def encode(self, sentence : str) -> list[int]:

        sentence = sentence.replace(" ", "▁")
        word_list = self.pattern.findall(sentence)

        buffer_word_list = []

        for word in word_list :
            
            
            word = list(word.encode())
            while True :
                
                min_id = 32001
                left = 0
                right = 0

                for i in range (len(word)-1):
                    if (word[i],word[i+1]) in self.merges :
                        if self.merges[word[i],word[i+1]] < min_id :
                            min_id = self.merges[word[i],word[i+1]]
                            left = word[i]
                            right = word[i+1]
                        
                if (min_id == 32001):
                    break
                
                new_word =[]
                k = 0
                while k < len(word):
                    if k!=(len(word)-1):
                        if word[k] == left and word[k+1] == right :
                            new_word.append(min_id)
                            k+=2
                            continue
                        
                    new_word.append(word[k])
                    k+=1

                word = new_word
                
            buffer_word_list.extend(word)

        return buffer_word_list


    def decode(self, ids: list[int]) -> str:
        byte_sequence = b''.join([self.vocab[idx] for idx in ids])
        return byte_sequence.decode('utf-8', errors='replace').replace("▁", " ")
import re

class Encodeur():
    def __init__(self):
        self.vocab = {}

        with open('C:/Users/linkm/Documents/GitHub/IA/src/inverse_vocab.txt', 'r') as f:
            lignes = f.read().split(" ")
            for i in range (15106):
                self.vocab[(lignes[i*3 +1],lignes[i*3 +2])] = lignes[i*3]

    def encode(self):
        
        

    def 
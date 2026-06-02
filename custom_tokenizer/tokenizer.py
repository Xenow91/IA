import re
import os

class Tokenizer:
    def __init__(self, vocab_path: str = None, merges_path: str = None):
        
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        vocab_path = vocab_path or os.path.join(base_dir, "vocab.txt")
        merges_path = merges_path or os.path.join(base_dir, "merges.txt")
        
        self.vocab = {} 
        self.merges = {}
        self.pattern = re.compile(r'▁?[a-zA-Zà-ÿ]+|▁?\d+|▁?[^\w\s ]+')
        self.cache = {}
        
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
            
            # 1. Vérification du Cache (Accélération fulgurante O(1))
            if word in self.cache:
                buffer_word_list.extend(self.cache[word])
                continue
                
            # 2. Si mot inconnu, on le calcule
            ids = list(word.encode("utf-8"))

            while len(ids) >= 2:
                # zip(ids, ids[1:]) crée les paires à la vitesse du C
                pairs = list(zip(ids, ids[1:]))

                # min() avec un dict.get permet de trouver la meilleure paire instantanément
                # Si la paire n'est pas dans self.merges, get retourne l'infini
                best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

                # Si la meilleure paire a un score infini, on ne peut plus rien fusionner
                if best_pair not in self.merges:
                    break

                new_id = self.merges[best_pair]

                # Remplacement efficace en 1 seule passe
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1

                ids = new_ids

            # 3. Sauvegarde dans le cache pour les prochaines fois
            self.cache[word] = ids
            
            # 4. Sécurité RAM : Si le cache devient trop gros (mots bizarres, typos), on le purge
            if len(self.cache) > 1_500_000:
                self.cache.clear()

            buffer_word_list.extend(ids)

        return buffer_word_list

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b''.join([self.vocab.get(idx, b'') for idx in ids])
        return byte_sequence.decode('utf-8', errors='replace').replace("▁", " ")
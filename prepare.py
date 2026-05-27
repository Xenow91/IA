import os
import numpy as np
from datasets import load_dataset
import tqdm
import tokenizer

def prepare_dataset():

    tok = tokenizer.Tokenizer("vocab.txt", "merges.txt")

    dataset = load_dataset("parquet", data_files="data/raw/fineweb.parquet", split="train")
    
    split_dataset = dataset.train_test_split(test_size=0.01, seed=42)
    split_dataset['val'] = split_dataset.pop('test')

    dtype = np.uint16

    eot_token = 32001 

    for split, dset in split_dataset.items():
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    
        # On ouvre le fichier en mode "Append Binary" (ab)
        with open(filename, 'ab') as f:

            buffer = []
            buffer_size_limit = 1_000_000 
            total_tokens = 0
            
            for example in tqdm.tqdm(dset, desc=f"Encodage {split}"):
                text = example['text']

                ids = tok.encode(text)
                
                buffer.extend(ids)
                buffer.append(eot_token)
                
                if len(buffer) >= buffer_size_limit:
                    total_tokens += len(buffer)

                    f.write(np.array(buffer, dtype=dtype).tobytes())
                    buffer = [] 
            
            if len(buffer) > 0:
                total_tokens += len(buffer)
                f.write(np.array(buffer, dtype=dtype).tobytes())
                
        print(f"Terminé pour {split} ! Total tokens : {total_tokens:,}")

if __name__ == "__main__":
    prepare_dataset()
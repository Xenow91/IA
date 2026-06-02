import os
import numpy as np
from datasets import load_dataset
import tqdm
from custom_tokenizer.tokenizer import Tokenizer

def prepare_dataset():
    # Initialisation de ton tokenizer
    tok = Tokenizer()

    # Chargement en streaming direct depuis HuggingFace
    print("Connexion au flux FineWeb-Edu...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    # Tes constantes (Validées pour l'empreinte RAM/Disque)
    dtype = np.uint16
    eot_token = 32001 
    buffer_size_limit = 1_000_000 

    train_filename = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_filename = os.path.join(os.path.dirname(__file__), 'val.bin')

    # On utilise deux buffers distincts pour gérer le split à la volée
    buffer_train = []
    buffer_val = []
    total_tokens_train = 0
    total_tokens_val = 0

    # Ouverture simultanée des deux fichiers
    with open(train_filename, 'ab') as f_train, open(val_filename, 'ab') as f_val:
        
        # Tqdm ne peut pas connaître le total en streaming, il affichera les itérations/s
        pbar = tqdm.tqdm(desc="Encodage FineWeb-Edu (Streaming)")
        
        for i, example in enumerate(dataset):
            text = example['text']
            ids = tok.encode(text)
            
            # Ajout du token de séparation
            ids.append(eot_token)
            
            # Routage : 1 document sur 100 (1%) part dans le set de validation
            is_val = (i % 100 == 0)
            
            if is_val:
                buffer_val.extend(ids)
                if len(buffer_val) >= buffer_size_limit:
                    total_tokens_val += len(buffer_val)
                    f_val.write(np.array(buffer_val, dtype=dtype).tobytes())
                    buffer_val = []  # Vidage du buffer
            else:
                buffer_train.extend(ids)
                if len(buffer_train) >= buffer_size_limit:
                    total_tokens_train += len(buffer_train)
                    f_train.write(np.array(buffer_train, dtype=dtype).tobytes())
                    buffer_train = [] # Vidage du buffer
            
            pbar.update(1)
        
        # Fin du flux : Flush final des buffers restants (très important pour ne rien perdre)
        if len(buffer_train) > 0:
            total_tokens_train += len(buffer_train)
            f_train.write(np.array(buffer_train, dtype=dtype).tobytes())
            
        if len(buffer_val) > 0:
            total_tokens_val += len(buffer_val)
            f_val.write(np.array(buffer_val, dtype=dtype).tobytes())
            
        pbar.close()
        
        print("\n--- Encodage Terminé ---")
        print(f"Tokens Train : {total_tokens_train:,}")
        print(f"Tokens Val   : {total_tokens_val:,}")

if __name__ == "__main__":
    prepare_dataset()
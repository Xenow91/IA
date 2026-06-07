import os
import numpy as np
from datasets import load_dataset
import tqdm
import tiktoken

def prepare_dataset():
    tok = tiktoken.get_encoding("gpt2")

    # Définition des chemins
    base_dir = os.path.dirname(__file__)
    train_filename = os.path.join(base_dir, 'train.bin')
    val_filename = os.path.join(base_dir, 'val.bin')
    state_filename = os.path.join(base_dir, 'resume_state.txt')

    # --- LOGIQUE DE REPRISE (RESILIENCE) ---
    start_idx = 0
    if os.path.exists(state_filename):
        with open(state_filename, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                start_idx = int(content)
                print(f"Fichier d'état détecté. Reprise à partir du document n°{start_idx}...")
    else:
        print("Aucun état précédent. Démarrage depuis le début.")

    print("Connexion au flux FineWeb-Edu...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # On avance le flux jusqu'au point de sauvegarde si nécessaire
    if start_idx > 0:
        dataset = dataset.skip(start_idx)

    dtype = np.uint16
    eot_token = tok.eot_token # 50256
    buffer_size_limit = 1_000_000 

    buffer_train = []
    buffer_val = []
    total_tokens_train = 0
    total_tokens_val = 0

    # On utilise 'ab' (Append Binary). Si les fichiers existent, on écrit à la suite.
    with open(train_filename, 'ab') as f_train, open(val_filename, 'ab') as f_val:
        
        pbar = tqdm.tqdm(desc="Encodage FineWeb-Edu", initial=start_idx)
        
        # enumerate(..., start=start_idx) permet de garder le vrai numéro du document
        for i, example in enumerate(dataset, start=start_idx):
            text = example['text']
            ids = tok.encode(text, disallowed_special=())
            ids.append(eot_token)
            
            is_val = (i % 100 == 0)
            
            if is_val:
                buffer_val.extend(ids)
            else:
                buffer_train.extend(ids)
                
            # --- CHECKPOINT ATOMIQUE ---
            # On déclenche la sauvegarde quand le gros buffer (train) est plein
            if len(buffer_train) >= buffer_size_limit:
                # 1. Vidage Train
                total_tokens_train += len(buffer_train)
                f_train.write(np.array(buffer_train, dtype=dtype).tobytes())
                buffer_train = []
                
                # 2. Vidage Val (pour garder les deux fichiers parfaitement synchronisés)
                if len(buffer_val) > 0:
                    total_tokens_val += len(buffer_val)
                    f_val.write(np.array(buffer_val, dtype=dtype).tobytes())
                    buffer_val = []
                    
                # 3. Sauvegarde de l'état (index actuel)
                with open(state_filename, 'w') as f_state:
                    f_state.write(str(i))
            
            pbar.update(1)
        
        # --- FLUSH FINAL (Fin du dataset) ---
        if len(buffer_train) > 0:
            total_tokens_train += len(buffer_train)
            f_train.write(np.array(buffer_train, dtype=dtype).tobytes())
        if len(buffer_val) > 0:
            total_tokens_val += len(buffer_val)
            f_val.write(np.array(buffer_val, dtype=dtype).tobytes())
            
        # Si on arrive à la toute fin, on peut supprimer le fichier d'état
        if os.path.exists(state_filename):
            os.remove(state_filename)
            
        pbar.close()
        
        print("\n--- Encodage Totalement Terminé ---")
        print(f"Tokens Train ajoutés : {total_tokens_train:,}")
        print(f"Tokens Val ajoutés   : {total_tokens_val:,}")

if __name__ == "__main__":
    prepare_dataset()
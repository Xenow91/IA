import os
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def prepare_sft():
    print("Téléchargement du dataset Dolly-15k...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    enc = tiktoken.get_encoding("gpt2")
    
    sft_data = []
    
    print("Formatage et Tokenization...")
    for item in tqdm(dataset):
        instruction = item['instruction']
        context = item.get('context', '')
        response = item['response']
        
        # Formatage du prompt (partie que le modèle lira en entrée)
        if context.strip() == "":
            prompt_text = f"Question: {instruction}\nAnswer: "
        else:
            prompt_text = f"Context: {context}\nQuestion: {instruction}\nAnswer: "
            
        # Formatage de la réponse (partie que le modèle doit prédire)
        response_text = f"{response}<|endoftext|>"
        
        # On autorise le special_token <|endoftext|>
        prompt_tokens = enc.encode(prompt_text, allowed_special={'<|endoftext|>'})
        response_tokens = enc.encode(response_text, allowed_special={'<|endoftext|>'})
        
        # Concaténation totale
        input_ids = prompt_tokens + response_tokens
        
        # IMPORTANT : On masque la Loss sur le Prompt. 
        # Ton fichier model.py utilise ignore_index=-1 (et non -100).
        labels = [-1] * len(prompt_tokens) + response_tokens
        
        # On filtre les exemples trop longs pour notre contexte
        if len(input_ids) <= 2048:
            sft_data.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)
            })
            
    print(f"Total d'exemples valides : {len(sft_data)}")
    
    os.makedirs("data", exist_ok=True)
    save_path = "data/sft_data.pt"
    torch.save(sft_data, save_path)
    print(f"Données de Fine-Tuning sauvegardées dans {save_path} !")

if __name__ == '__main__':
    prepare_sft()

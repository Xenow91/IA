import torch
from model import GPT, GPTConfig 
from tokenizer import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Chargement sur : {device}")

config = GPTConfig(vocab_size=32064, block_size=1024)
model = GPT(config)
model.to(device)

# On charge le fichier directement depuis ton disque dur local
checkpoint = torch.load('data/best_model.pt', map_location=device, weights_only=False)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()

tok = Tokenizer("data/vocab.txt", "data/merges.txt")
print("Modèle prêt pour l'inférence !")


while True:
    prompt = input("\nToi : ")
    if prompt.lower() in ['quit', 'exit', 'stop']:
        break
        
    ids = tok.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y = model.generate(idx=x, max_new_tokens=100, temperature=0.8, top_k=50)
        
    output_ids = y[0].tolist()

    if 32001 in output_ids:
        stop_index = output_ids.index(32001)
        output_ids = output_ids[:stop_index]

    output_text = tok.decode(output_ids)
    
    print(f"\nIA : {output_text}")
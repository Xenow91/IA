import os
import torch
import numpy as np
from model import GPT, GPTConfig 

class DataLoaderLite:
    def __init__(self, data_dir, split, batch_size, block_size, device):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        filename = os.path.join(data_dir, f'{split}.bin')
        
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        print(f"[{split}] DataLoader initialisé. Total tokens : {len(self.data):,}")

    def get_batch(self):
        """
        Génère un lot (batch) de tenseurs X (entrées) et Y (cibles) et les envoie sur le GPU.
        """

        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,))
        

        x = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        
        if 'cuda' in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Entraînement sur : {device}")
      
block_size = 1024     
max_iters = 15000      
learning_rate = 3e-4 

micro_batch_size = 2  
grad_accum_steps = 8     


batch_size = micro_batch_size

data_dir = "data" 
train_loader = DataLoaderLite(data_dir=data_dir, split="train", 
                              batch_size=batch_size, block_size=block_size, 
                              device=device)

val_loader = DataLoaderLite(data_dir=data_dir, split="val", 
                              batch_size=batch_size, block_size=block_size, 
                              device=device)

config = GPTConfig(vocab_size=32064, block_size=block_size, n_head=16, n_embd=1024, n_layer=16 )
model = GPT(config)
model.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    out = {}
    model.eval() 
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        
    model.train() 
    return out


start_iter = 0
best_val_loss = float('inf')

checkpoint_path = 'ckpt.pt'


if os.path.exists(checkpoint_path):
    print(f"Fichier de sauvegarde détecté. Reprise de l'entraînement depuis {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    best_val_loss = checkpoint['best_val_loss']

    start_iter = checkpoint['iter_num'] + 1
    print(f"-> Reprise configurée à l'itération {start_iter} (Meilleure Val Loss connue : {best_val_loss:.4f})")
else:
    print("Aucun checkpoint trouvé. Démarrage d'un nouvel entraînement à partir de zéro.")

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, 
                                       betas=(0.9, 0.95), device_type=device)

if os.path.exists(checkpoint_path):
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("-> État de l'optimiseur AdamW restauré avec succès.")

# print("Compilation du modèle")
# model = torch.compile(model)

for iter in range(start_iter, max_iters):
    
    if iter % 500 == 0: 
        losses = estimate_loss(model, eval_iters=20)
        print(f"Itération {iter} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}", flush=True)
  
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter,
            'best_val_loss': best_val_loss,
            'config': config 
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"--> Checkpoint de sécurité sauvegardé (Itération {iter})", flush=True)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint['best_val_loss'] = best_val_loss
            best_checkpoint_path = checkpoint_path.replace('ckpt.pt', 'best_model.pt')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"--> Nouveau record détecté ! Modèle copié dans {best_checkpoint_path}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
            
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        loss.backward()
        
    optimizer.step()
    
    if iter % 10 == 0:
        print(f"Step {iter:04d} | Effective Batch Loss: {loss_accum.item():.4f}", flush=True)

    
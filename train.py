import os
import torch
import numpy as np
from model import GPT, GPTConfig 
import math
import wandb # Ajout de Weights & Biases

# --- OPTIMISATIONS MATÉRIELLES ---
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DataLoaderLite:
    def __init__(self, data_dir, split, batch_size, block_size, device):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.split = split 
        
        filename = os.path.join(data_dir, f'{split}.bin')
        
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        self.current_position = 0
        
        print(f"[{self.split}] DataLoader initialisé. Total tokens : {len(self.data):,}")

    def get_batch(self):
        
        if self.current_position + (self.batch_size * self.block_size + 1) > len(self.data):
            self.current_position = 0
            print(f"\n--- Fin du dataset {self.split} atteinte. Début de l'Epoch suivante ---")

        buf = self.data[self.current_position : self.current_position + self.batch_size * self.block_size + 1]
        buf_tensor = torch.from_numpy(buf.astype(np.int32))
        
        x = buf_tensor[:-1].view(self.batch_size, self.block_size)
        y = buf_tensor[1:].view(self.batch_size, self.block_size)
        
        self.current_position += self.batch_size * self.block_size
        
        if 'cuda' in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Entraînement sur : {device}")
      
block_size = 2048

max_lr = 3e-4
min_lr = 3e-5
warmup_steps = 1000
max_iters = 20_000

# Paramètres de Batch mis à jour (Optimisés 5060 Ti - 16Go)
micro_batch_size = 2  
grad_accum_steps = 128    
batch_size = micro_batch_size

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

data_dir = "data" 
train_loader = DataLoaderLite(data_dir=data_dir, split="train", 
                              batch_size=batch_size, block_size=block_size, 
                              device=device)

val_loader = DataLoaderLite(data_dir=data_dir, split="val", 
                              batch_size=batch_size, block_size=block_size, 
                              device=device)

config = GPTConfig(
    vocab_size=50304, 
    block_size=block_size, 
    n_head=16, 
    n_embd=1024, 
    n_layer=26,
    n_kv_heads=4,
    use_checkpointing=False
)
model = GPT(config)
model.to(device)

if 'cuda' in device:
    print("Compilation du modèle avec torch.compile...")
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    out = {}
    model.eval() 
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch()
            
            # Utilisation de l'autocast ici pour l'évaluation (bfloat16)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
    
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(checkpoint['model'])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    start_iter = checkpoint.get('iter_num', 0) + 1
    print(f"-> Reprise configurée à l'itération {start_iter} (Meilleure Val Loss connue : {best_val_loss:.4f})")
else:
    print("Aucun checkpoint trouvé. Démarrage d'un nouvel entraînement à partir de zéro.")

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, 
                                       betas=(0.9, 0.95), device_type=device)

if os.path.exists(checkpoint_path):
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("-> État de l'optimiseur AdamW restauré avec succès.")

# Initialisation de Weights & Biases
wandb_log = True
if wandb_log:
    wandb.init(
        project="gros-model", 
        name="training-run",
        resume="allow",
        config={
            "batch_size_global": micro_batch_size * grad_accum_steps,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_iters": max_iters,
            "learning_rate": max_lr,
            "block_size": block_size,
            "vocab_size": config.vocab_size,
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "bfloat16": True
        }
    )

for iter in range(start_iter, max_iters):
    
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Sauvegarde et Évaluation tous les 500 steps (comme demandé : save_steps=500)
    if iter > 0 and iter % 500 == 0: 
        losses = estimate_loss(model, eval_iters=20)
        print(f"Itération {iter} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e}", flush=True)
  
        if wandb_log:
            wandb.log({
                "val/loss": losses['val'],
                "val/train_loss": losses['train']
            }, step=iter)

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

    # Mise à zéro des gradients
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    # Accumulation de gradients
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        
        # Sécurité numérique et performance : utilisation de bfloat16
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
            
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        loss.backward()
        
    # Mise à jour des poids
    optimizer.step()
    
    # Affichage et log W&B tous les 10 steps
    if iter % 10 == 0:
        print(f"Step {iter:06d} | Effective Batch Loss: {loss_accum.item():.4f} | LR: {lr:.2e}", flush=True)
        if wandb_log:
            wandb.log({
                "train/loss": loss_accum.item(),
                "train/lr": lr,
            }, step=iter)

if wandb_log:
    wandb.finish()
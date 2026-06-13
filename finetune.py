import os
import torch
import numpy as np
import math
from model import GPT, GPTConfig 
import random
import wandb

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SFTDataLoader:
    def __init__(self, data_path, batch_size, block_size, device):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        print(f"Chargement des données SFT depuis {data_path}...")
        self.data = torch.load(data_path)
        random.shuffle(self.data)
        self.current_idx = 0
        print(f"Dataloader SFT initialisé. Total exemples : {len(self.data)}")

    def get_batch(self):
        batch_input_ids = []
        batch_labels = []
        
        for _ in range(self.batch_size):
            if self.current_idx >= len(self.data):
                self.current_idx = 0
                random.shuffle(self.data) # On remélange à chaque nouvelle époque
                print("\n--- Fin du dataset SFT atteinte. Début de l'Epoch suivante ---")
                
            sample = self.data[self.current_idx]
            self.current_idx += 1
            
            input_ids = sample['input_ids'].tolist()
            labels = sample['labels'].tolist()
            
            # Padding : on remplit avec des tokens vides pour atteindre block_size
            if len(input_ids) < self.block_size:
                pad_len = self.block_size - len(input_ids)
                input_ids = input_ids + [50256] * pad_len # endoftext
                labels = labels + [-1] * pad_len          # Ignoré par la Loss
            else:
                input_ids = input_ids[:self.block_size]
                labels = labels[:self.block_size]
                
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_labels, dtype=torch.long)
        
        if 'cuda' in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Supervised Fine-Tuning sur : {device}")
      
block_size = 2048

# Hyperparamètres spécifiques au SFT (Très doux)
max_lr = 2e-5      # 10x plus petit que le pré-entraînement
min_lr = 2e-6
warmup_steps = 100
max_iters = 1500   # ~1.5 Epochs sur Dolly-15k, suffisant pour l'instruction-tuning

micro_batch_size = 2  
grad_accum_steps = 8   # Batch global de 16, largement suffisant pour le SFT
batch_size = micro_batch_size

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)

data_path = "data/sft_data.pt" 
train_loader = SFTDataLoader(data_path=data_path, batch_size=batch_size, block_size=block_size, device=device)

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

# CHARGEMENT OBLIGATOIRE DU MODÈLE PRÉ-ENTRAÎNÉ
checkpoint_path = 'best_model.pt'
sft_checkpoint_path = 'sft_model.pt'

if os.path.exists(checkpoint_path):
    print(f"Chargement du modèle pré-entraîné depuis {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.load_state_dict(checkpoint['model'])
    print(f"-> Modèle de base chargé avec succès !")
else:
    print(f"ERREUR CRITIQUE : {checkpoint_path} introuvable.")
    print("Vous devez d'abord télécharger best_model.pt à la fin de votre pré-entraînement Vast.ai !")
    exit()

# Nouvel optimiseur tout neuf pour cette nouvelle phase
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, 
                                       betas=(0.9, 0.95), device_type=device)

wandb_log = True
if wandb_log:
    wandb.init(
        project="gros-model", 
        name="sft-instruction-tuning",
        config={
            "batch_size_global": micro_batch_size * grad_accum_steps,
            "max_iters": max_iters,
            "learning_rate": max_lr,
            "type": "Supervised Fine-Tuning"
        }
    )

start_iter = 0
for iter in range(start_iter, max_iters):
    
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Sauvegarde régulière
    if iter > 0 and iter % 200 == 0: 
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        checkpoint_sft = {
            'model': raw_model.state_dict(),
            'iter_num': iter,
            'config': config 
        }
        torch.save(checkpoint_sft, sft_checkpoint_path)
        print(f"--> Checkpoint SFT sauvegardé (Itération {iter})", flush=True)

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
        print(f"Step {iter:06d} | SFT Loss: {loss_accum.item():.4f} | LR: {lr:.2e}", flush=True)
        if wandb_log:
            wandb.log({
                "sft/train_loss": loss_accum.item(),
                "sft/lr": lr,
            }, step=iter)

if wandb_log:
    wandb.finish()

# Sauvegarde de l'assistant final
raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
torch.save({'model': raw_model.state_dict(), 'config': config}, sft_checkpoint_path)
print(f"\n🎉 Entraînement SFT terminé ! Ton modèle Assistant est sauvegardé dans : {sft_checkpoint_path}")

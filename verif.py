import torch

# Charge uniquement le dictionnaire (pas besoin du modèle complet)
checkpoint = torch.load('data/ckpt.pt', map_location='cpu', weights_only=False)

print(f"Itération atteinte : {checkpoint['iter_num']}")
print(f"Meilleure Loss : {checkpoint['best_val_loss']:.4f}")
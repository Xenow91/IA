# Projet IA : Modèle de Langage de 450M Paramètres de Zéro

Ce projet contient l'intégralité du code source pour créer, pré-entraîner et affiner (Fine-Tuning) un modèle de langage (LLM) de type GPT depuis zéro, en utilisant PyTorch. Pour l'essayer : https://xenow91.github.io/vitrine/index.html 

## Objectif du Projet
L'objectif est de démontrer qu'il est possible d'entraîner un modèle de 450 Millions de paramètres sur 12 Milliards de tokens avec du matériel grand public (une simple RTX 5060 Ti 16 Go de VRAM) via la plateforme Vast.ai, en appliquant les techniques de l'état de l'art (FlashAttention, bfloat16, RMSNorm, RoPE).

## Caractéristiques du Modèle
- Architecture : Transformer
- Paramètres : ~446 Millions
- Couches (Layers) : 26
- Têtes d'attention (Heads) : 16
- Dimension de l'Embedding (d_model) : 1024
- Taille du Vocabulaire : 50 304 tokens
- Contexte maximum : 2048 tokens

## Optimisations Techniques Implémentées
Le code intègre des optimisations modernes pour maximiser la vitesse et réduire l'empreinte mémoire :
1. FlashAttention (SDP) : Accélération matérielle du calcul de l'attention.
2. Mixed Precision (bfloat16) : Division par deux de la consommation VRAM sans perte de stabilité.
3. Gradient Accumulation : Simulation d'un batch global de 256 séquences pour stabiliser la descente de gradient malgré la limite des 16 Go de VRAM.
4. Fused AdamW & Cosine Decay : Optimiseur fusionné en C++ et baisse dynamique du Learning Rate pour converger vers la meilleure Loss.
5. Rotary Positional Embeddings (RoPE) et RMSNorm pour une meilleure stabilité mathématique.

## Structure du Projet

### 1. Tokenizer et Préparation des Données
À des fins pédagogiques, un tokenizer personnalisé (`custom_tokenizer`) a été développé pour comprendre la mécanique sous-jacente du Byte-Pair Encoding (BPE). Cependant, pour des raisons de performance et de standardisation, le projet final utilise le tokenizer optimisé d'OpenAI (`tiktoken` version GPT-2).
Le script `prepare.py` gère le téléchargement et la tokenization d'un sous-ensemble du dataset HuggingFaceFW/fineweb-edu (textes éducatifs de haute qualité), générant les fichiers binaires `train.bin` et `val.bin` pour une lecture hyper-rapide via `np.memmap`.

### 2. Architecture du Modèle (model.py)
Définition complète du réseau de neurones avec PyTorch. Pas de bibliothèques tierces cachées, toute l'architecture (Attention, FeedForward, Normalisation) est écrite "from scratch".

### 3. Pré-Entraînement (train.py)
Script d'entraînement intensif conçu pour tourner sur Vast.ai. Intègre un suivi en direct sur WandB et un système de sauvegarde automatique du meilleur checkpoint (`best_model.pt`) chaque fois qu'un record de Loss est battu.

### 4. Instruction Tuning / SFT (prepare_sft.py & finetune.py)
Une fois le modèle doté de la parole (Pre-training), ces scripts le transforment en Assistant.
- Téléchargement du dataset Databricks Dolly-15k.
- Application d'un Loss Masking (`ignore_index=-1`) sur les instructions de l'utilisateur pour forcer le modèle à apprendre uniquement sur les réponses.
- Apprentissage lent (Low Learning Rate) sur 1.5 Epoch pour éviter l'oubli catastrophique (Catastrophic Forgetting).

## Reproduction de l'entraînement
1. Installez les dépendances (`torch`, `numpy`, `datasets`, `tiktoken`, `wandb`).
2. Lancez `python prepare.py` pour télécharger les 20 Go de données de pré-entraînement.
3. Transférez le dossier sur votre machine GPU (ex: Vast.ai) via SFTP/WinSCP.
4. Lancez `python train.py`. L'entraînement prend environ 9 jours sur une RTX 5060 Ti.
5. À la fin, exécutez `python prepare_sft.py` puis `python finetune.py` pour la phase d'instruction-tuning.

---
Projet réalisé à des fins d'apprentissage et de recherche sur les LLMs.

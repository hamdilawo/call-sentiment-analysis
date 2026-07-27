"""
main.py - Pipeline principal d'entraînement et de comparaison des 3 modèles CNN
Devoir Pratique Deep Learning - Classification binaire sur DIT-FLOWER_DATASET

Modèles comparés :
    1. LeNet-5 (Tanh / ReLU / Sigmoid) — from scratch — 32×32
    2. VGG16                           — from scratch — 224×224
    3. ResNet18                        — fine-tuning  — 224×224

Usage :
    python main.py --model all          # Lance tout (≈ 2-3h avec GPU)
    python main.py --model lenet        # Seulement LeNet-5 (3 variantes)
    python main.py --model vgg16
    python main.py --model resnet18
    python main.py --model lenet --epochs 10 --no-wandb  # Test rapide
"""

import argparse
import os
import torch
import torch.nn as nn
import wandb

from dataset import CustomImageDataset, get_transforms
from models.lenet5   import LeNet5
from models.vgg16    import VGG16
from models.resnet18 import ResNet18FineTuned
from train import train_model
from utils import set_seed, inspect_dataset, get_dataloaders, print_model_summary, compare_models

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DES CHEMINS — adapter selon votre environnement
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\lenovo thinkbook\Documents\Master2\DeepLearning2\dv2\DIT-FLOWER_DATASET\ANSD-FLOWER_DATASET"
TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
VAL_DIR    = os.path.join(DATA_ROOT, "valid")
MODELS_DIR = "checkpoints"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES DE RÉFÉRENCE (section 5 du devoir)
# ─────────────────────────────────────────────────────────────────────────────
HPARAMS = {
    'lenet': {
        'lr':         1e-3,
        'batch_size': 64,
        'epochs':     50,
        'optimizer':  'adam',
        'input_size': 32,
        'seed':       42,
    },
    'vgg16': {
        'lr':         1e-4,
        'batch_size': 16,   # VRAM limitée
        'epochs':     30,
        'optimizer':  'adam',
        'input_size': 224,
        'seed':       42,
    },
    'resnet18': {
        'lr':         1e-3,
        'batch_size': 32,
        'epochs':     50,
        'optimizer':  'adam',
        'input_size': 224,
        'seed':       42,
    },
}


def get_device() -> torch.device:
    """Sélectionne automatiquement le meilleur dispositif disponible."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU détecté : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Aucun GPU détecté — entraînement sur CPU (lent)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE LeNet-5 (3 variantes d'activation)
# ─────────────────────────────────────────────────────────────────────────────

def run_lenet(device, use_wandb: bool = True, epochs_override: int = None):
    """
    Entraîne les 3 variantes de LeNet-5 et compare les activations.
    Chaque variante est loguée dans un run wandb séparé.
    """
    hp = HPARAMS['lenet']
    epochs = epochs_override or hp['epochs']

    print("\n" + "="*60)
    print("  MODÈLE 1 : LeNet-5 (3 variantes d'activation)")
    print("="*60)

    # Datasets LeNet (32×32, sans normalisation ImageNet)
    train_tf = get_transforms('train', model_type='lenet')
    val_tf   = get_transforms('val',   model_type='lenet')

    train_ds = CustomImageDataset(TRAIN_DIR, transform=train_tf, mode='train')
    val_ds   = CustomImageDataset(VAL_DIR,   transform=val_tf,   mode='val')
    class_names = train_ds.classes

    lenet_histories = {}

    for activation in ['tanh', 'relu', 'sigmoid']:
        run_name = f"lenet5_{activation}"
        print(f"\n{'─'*50}")
        print(f"  Variante : LeNet5-{activation.upper()}")
        print(f"{'─'*50}")

        # Seed identique pour toutes les variantes (comparaison équitable)
        set_seed(hp['seed'])

        # DataLoaders (recréés à chaque variante pour garantir le même ordre)
        train_loader, val_loader = get_dataloaders(
            train_ds, val_ds,
            batch_size=hp['batch_size']
        )

        # Modèle
        model = LeNet5(num_classes=2, activation=activation).to(device)
        print_model_summary(model, run_name)

        # Optimiseur et loss
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
        criterion = nn.CrossEntropyLoss()

        # Scheduler (optionnel) : réduit le LR si la val_loss stagne
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )

        # Initialisation wandb
        if use_wandb:
            wandb.init(
                project="devoir-cnn-dit",
                name=run_name,
                config={**hp, 'activation': activation, 'model': 'lenet5', 'epochs': epochs},
            )

        # Entraînement
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=epochs,
            model_name=run_name,
            save_path=os.path.join(MODELS_DIR, f"best_{run_name}.pth"),
            scheduler=scheduler,
            class_names=class_names,
        )

        lenet_histories[run_name] = history

        if use_wandb:
            wandb.finish()

    # Comparaison des 3 courbes
    compare_models(lenet_histories, metric='val_acc')
    compare_models(lenet_histories, metric='val_loss')

    return lenet_histories


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE VGG16
# ─────────────────────────────────────────────────────────────────────────────

def run_vgg16(device, use_wandb: bool = True, epochs_override: int = None):
    """Entraîne VGG16 from scratch."""
    hp = HPARAMS['vgg16']
    epochs = epochs_override or hp['epochs']

    print("\n" + "="*60)
    print("  MODÈLE 2 : VGG16 From Scratch")
    print("="*60)

    set_seed(hp['seed'])

    train_tf = get_transforms('train', model_type='standard')
    val_tf   = get_transforms('val',   model_type='standard')

    train_ds = CustomImageDataset(TRAIN_DIR, transform=train_tf, mode='train')
    val_ds   = CustomImageDataset(VAL_DIR,   transform=val_tf,   mode='val')
    class_names = train_ds.classes

    train_loader, val_loader = get_dataloaders(
        train_ds, val_ds,
        batch_size=hp['batch_size']
    )

    model = VGG16(num_classes=2, dropout_p=0.5).to(device)
    print_model_summary(model, 'VGG16 From Scratch')

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if use_wandb:
        wandb.init(
            project="devoir-cnn-dit",
            name="vgg16_scratch",
            config={**hp, 'model': 'vgg16', 'epochs': epochs},
        )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        model_name='vgg16_scratch',
        save_path=os.path.join(MODELS_DIR, "best_vgg16.pth"),
        scheduler=scheduler,
        class_names=class_names,
    )

    if use_wandb:
        wandb.finish()

    return history


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ResNet18
# ─────────────────────────────────────────────────────────────────────────────

def run_resnet18(device, use_wandb: bool = True, epochs_override: int = None):
    """Entraîne ResNet18 par transfer learning (feature extraction)."""
    hp = HPARAMS['resnet18']
    epochs = epochs_override or hp['epochs']

    print("\n" + "="*60)
    print("  MODÈLE 3 : ResNet18 Fine-tuning (Transfer Learning)")
    print("="*60)

    set_seed(hp['seed'])

    train_tf = get_transforms('train', model_type='standard')
    val_tf   = get_transforms('val',   model_type='standard')

    train_ds = CustomImageDataset(TRAIN_DIR, transform=train_tf, mode='train')
    val_ds   = CustomImageDataset(VAL_DIR,   transform=val_tf,   mode='val')
    class_names = train_ds.classes

    train_loader, val_loader = get_dataloaders(
        train_ds, val_ds,
        batch_size=hp['batch_size']
    )

    # Phase 1 : Feature extraction (backbone gelé)
    model = ResNet18FineTuned(num_classes=2, freeze_backbone=True).to(device)
    print_model_summary(model, 'ResNet18 Fine-tuning')

    # N'optimiser que les paramètres de la tête (requires_grad=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp['lr']
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    if use_wandb:
        wandb.init(
            project="devoir-cnn-dit",
            name="resnet18_finetune",
            config={
                **hp,
                'model':             'resnet18',
                'strategy':          'feature_extraction',
                'backbone_frozen':   True,
                'epochs':            epochs,
            },
        )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        model_name='resnet18_finetune',
        save_path=os.path.join(MODELS_DIR, "best_resnet18.pth"),
        scheduler=scheduler,
        class_names=class_names,
    )

    if use_wandb:
        wandb.finish()

    return history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pipeline CNN — Devoir Deep Learning DIT')
    parser.add_argument('--model',    type=str, default='all',
                        choices=['all', 'lenet', 'vgg16', 'resnet18'],
                        help='Modèle à entraîner (défaut: all)')
    parser.add_argument('--epochs',   type=int, default=None,
                        help='Surcharge le nombre d\'époques (utile pour les tests)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Désactive le tracking wandb (mode debug)')
    parser.add_argument('--inspect',  action='store_true',
                        help='Inspecte le dataset avant l\'entraînement')
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    # ── Seed globale ──────────────────────────────────────────────────────────
    set_seed(42)

    # ── Dispositif ────────────────────────────────────────────────────────────
    device = get_device()

    # ── Inspection optionnelle du dataset ─────────────────────────────────────
    if args.inspect:
        print("\n--- Inspection train/ ---")
        inspect_dataset(TRAIN_DIR)
        print("\n--- Inspection val/ ---")
        inspect_dataset(VAL_DIR)

    # ── Connexion wandb ───────────────────────────────────────────────────────
    if use_wandb:
        # Remplacer 'votre_entity' par votre username wandb
        os.environ.setdefault('WANDB_PROJECT', 'devoir-cnn-dit')
        print("✓ Wandb configuré — projet : devoir-cnn-dit")

    # ── Lancement des entraînements ───────────────────────────────────────────
    all_histories = {}

    if args.model in ('all', 'lenet'):
        lenet_hist = run_lenet(device, use_wandb=use_wandb, epochs_override=args.epochs)
        all_histories.update(lenet_hist)

    if args.model in ('all', 'vgg16'):
        vgg_hist = run_vgg16(device, use_wandb=use_wandb, epochs_override=args.epochs)
        all_histories['vgg16_scratch'] = vgg_hist

    if args.model in ('all', 'resnet18'):
        resnet_hist = run_resnet18(device, use_wandb=use_wandb, epochs_override=args.epochs)
        all_histories['resnet18_finetune'] = resnet_hist

    # ── Comparaison finale des 3 meilleurs modèles ────────────────────────────
    if args.model == 'all' and len(all_histories) >= 3:
        print("\n\n=== COMPARAISON FINALE DES MODÈLES ===")
        best_per_model = {
            'lenet5_tanh':         all_histories.get('lenet5_tanh',    {}),
            'vgg16_scratch':       all_histories.get('vgg16_scratch',  {}),
            'resnet18_finetune':   all_histories.get('resnet18_finetune', {}),
        }
        compare_models(best_per_model, metric='val_acc')
        compare_models(best_per_model, metric='val_f1')

        # Tableau récapitulatif
        print(f"\n{'─'*60}")
        print(f"{'Modèle':<25} {'Meill. Val Acc':>15} {'Meill. Val F1':>15}")
        print(f"{'─'*60}")
        for name, hist in best_per_model.items():
            if hist:
                best_acc = max(hist.get('val_acc', [0]))
                best_f1  = max(hist.get('val_f1',  [0]))
                print(f"  {name:<23} {best_acc:>15.4f} {best_f1:>15.4f}")
        print(f"{'─'*60}")


if __name__ == '__main__':
    main()

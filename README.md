# NeuroTrace

**Détection et traçabilité sécurisée des tumeurs cérébrales par Intelligence Artificielle et Blockchain**

NeuroTrace est un prototype pédagogique qui combine l'intelligence artificielle et la blockchain pour analyser des images IRM cérébrales et assurer la traçabilité des différentes étapes du traitement.

Le projet utilise un modèle ResNet50 pour classifier les images IRM en quatre catégories :
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Objectifs

- Prétraiter les images IRM cérébrales.
- Effectuer une classification à l'aide d'un modèle d'intelligence artificielle.
- Évaluer le niveau de confiance de la prédiction.
- Valider les résultats selon des règles définies.
- Générer une explication visuelle avec Grad-CAM.
- Assurer progressivement la traçabilité et l'intégrité des artefacts grâce à SHA-256 et Hyperledger Fabric.

## Fonctionnalités développées

| Module | Description | État |
|---|---|---|
| Prétraitement | Validation, conversion RGB et redimensionnement des images | Développé |
| Intelligence artificielle | Intégration du modèle ResNet50 à quatre classes | Développé |
| Prédiction | Classe prédite et scores de confiance | Développé |
| Validation | Accepté, Révision requise ou Rejeté selon la confiance | Première version |
| Explicabilité | Génération de visualisations Grad-CAM | Développé |
| Sécurité SHA-256 | Hachage et vérification de l'intégrité | À développer |
| Blockchain | Traçabilité avec Hyperledger Fabric | À développer |
| Interface utilisateur | Interface de consultation et de démonstration | À développer |

## Architecture du projet

```text
NeuroTrace_AI/
├── config/
├── data/
├── datasets/
├── models/
├── scripts/
├── src/
│   └── neurotrace/
│       ├── preprocessing/
│       ├── ai/
│       ├── validation/
│       ├── explainability/
│       ├── storage/
│       ├── security/
│       ├── blockchain/
│       ├── agents/
│       └── pipeline/
├── tests/
├── artifacts/
├── docs/
├── requirements.txt
└── README.md
```

## Technologies utilisées

- Python 3.11
- PyTorch et Torchvision
- ResNet50
- Hugging Face Hub
- Pillow et OpenCV
- Grad-CAM
- Pytest

## Installation

### 1. Cloner le dépôt

```bash
git clone URL_DU_DEPOT
cd NeuroTrace_AI
```

Remplacer `URL_DU_DEPOT` par l'adresse HTTPS du dépôt GitHub.

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

Sous Windows PowerShell :

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Télécharger le modèle

Le modèle ResNet50 est téléchargé depuis Hugging Face lors de son premier chargement.

Dépôt du modèle :  
https://huggingface.co/Abuzaid01/brain-tumor-resnet50-classifier

Une connexion Internet est donc nécessaire lors du premier téléchargement.

## Tests

Depuis la racine du projet, exécuter :

```bash
python -m pytest tests/ -v
```

Les tests vérifient notamment le prétraitement, la prédiction, la validation et la génération Grad-CAM.

## Données

Les jeux de données IRM ne sont pas inclus dans le dépôt GitHub. Chaque membre de l'équipe doit obtenir le jeu de données utilisé pour le projet et le placer dans la structure attendue par les scripts.

## Limites du projet

NeuroTrace est un prototype pédagogique. Il ne constitue pas un dispositif médical certifié et ne doit pas être utilisé pour établir un diagnostic clinique autonome.

Les prédictions et les visualisations Grad-CAM sont expérimentales et ne remplacent pas l'interprétation d'un professionnel de santé.

## Équipe

Projet réalisé dans le cadre d'un projet académique.

## Licence

À définir par l'équipe.

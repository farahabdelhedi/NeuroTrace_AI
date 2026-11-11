
# 1. Image de base : Python 3.11
FROM python:3.11-slim

# 2. Dossier de travail dans le conteneur
WORKDIR /app


# Installer les bibliothèques système nécessaires à OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# 3. Configuration des chemins Python
ENV PYTHONPATH="/app/src:/app"

# 4. Copier le fichier des dépendances
COPY requirements.txt .

# 5. Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copier le code source et la configuration
COPY src/ ./src/
COPY config/ ./config/

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger tous les jeux de données NLTK nécessaires
RUN python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng

COPY . .

ENV PYTHONPATH=/app

CMD ["watchmedo", "auto-restart", "--directory=.", "--pattern=*.py", "--recursive", "--", "python", "app/main.py"]

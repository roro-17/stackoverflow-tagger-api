# utils.py
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from ast import literal_eval

_data = pd.read_csv("data/stack_questions_cleaned.csv", sep=";")
for col in ['Title', 'Body', 'Tags', 'text_comb']:
    _data[col] = _data[col].apply(literal_eval)

_mlb = MultiLabelBinarizer()
_mlb.fit(_data['Tags'])

def inv_transform(label):
    threshold = 0.12
    binary_labels = (label > threshold).astype(int)
    return _mlb.inverse_transform(binary_labels)

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    return {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }.get(tag, wordnet.NOUN)

def sentence_cleaner(text):
    text = text.lower()
    text = re.sub("\'\w+", ' ', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub('[^\w\s#\s++]', ' ', text)
    text = re.sub(r'\w*\d+\w*', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = BeautifulSoup(text, "lxml").text
    tokens = text.split()
    stop_words = stopwords.words('english')
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
# model.py
import joblib
import os
import numpy as np
from app.utils import inv_transform

MODEL_PATH = os.getenv("MODEL_PATH", "static/models/reg_logit_nlp.pkl")
lr_model = joblib.load(MODEL_PATH)

def predict_tags(text_tokens):
    y_pred = lr_model.predict_proba(np.array([text_tokens]))
    return inv_transform(y_pred)
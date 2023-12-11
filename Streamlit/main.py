import streamlit as st
import re
import nltk
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pickle
from tensorflow.keras import activations
import tensorflow as tf
import numpy as np


import sys
from pathlib import Path

dir = Path(__file__)
sys.path.append(dir.parent.parent)



nltk.download('wordnet')


#st.balloons()


def get_text():
    text = st.text_area('','Fish tank leaks')
    st.write(f'You wrote {len(text)} characters.')
    return text


def text_pre_processing(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    text = " ".join(lst_text)
    return text



def construct_encodings(x, tokenizer, max_len, trucation=True, padding=True):
    return tokenizer(x, max_length=max_len, truncation=trucation, padding=padding)
    

def construct_tfdataset(encodings, y=None):
    if y:
        return tf.data.Dataset.from_tensor_slices((dict(encodings),y))
    else:
        return tf.data.Dataset.from_tensor_slices(dict(encodings))
    

def create_predictor(model, model_name, max_len):
  tkzr = DistilBertTokenizer.from_pretrained(model_name)
  def predict_proba(text):
      x = [text]

      encodings = construct_encodings(x, tkzr, max_len=max_len)
      tfdataset = construct_tfdataset(encodings)
      tfdataset = tfdataset.batch(1)

      preds = model.predict(tfdataset).logits
      preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
      return  np.argmax(preds, axis=1)
    
  return predict_proba



def main():
    st.sidebar.subheader("Amazon Pets Product Reviews")
    st.title("Sentiment Analysis")
    st.divider()
    st.subheader("Step 1: Leave your reivew here")

    text = get_text()

    if st.button('Submit', type="primary"):
        st.divider()
        st.subheader("Step 2: Clean the reivew text")
        text = text_pre_processing(text)
        st.write(text)

        st.divider()
        st.subheader("Step 3: Return the sentiment of the review")

        new_model = TFDistilBertForSequenceClassification.from_pretrained('model/clf')
        model_name, max_len = pickle.load(open('model/info.pkl', 'rb'))
        clf = create_predictor(new_model, model_name, max_len)

        result = clf(text)
     
        sentiment = 'Negative' if result == 0 else 'Positive'

        st.write(f"The sentiment of the review is: ***{sentiment}***")

if __name__ == "__main__":
    main()


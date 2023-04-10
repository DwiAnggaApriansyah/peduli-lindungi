# peduli-lindungi

import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io
import pandas as pd

data = files.upload()
data = pd.read_csv('pedulilindungi.csv')
data.head()
Casefolding
import re
def casefolding(Review):
  Review = Review.lower()
  Review = Review.strip(" ")
  Review = re.sub(r'[?|$|.|!2_:")(-+,]','', Review)
  return Review
data['Review'] = data['Review'].apply(casefolding)
data.head(100)
Tokenizing
def token(Review):
  nstr = Review.split(' ')
  dat = []
  a = -1
  for hu in nstr:
    a = a + 1
  if hu == '':
    dat.append(a)
    p = 0
    b = 0
  for q in dat:
      b = q - p
      del nstr[b]
      p = p + 1
  return nstr
data['Review'] = data['Review'].apply(token)
data.head(10)
Filtering
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def stopword_removal(Review):
  filtering = stopwords.words('indonesian','english')
  x = []
  data = []
  def myFunc(x):
    if x in filtering:
      return False
    else:
      return True
  fit = filter(myFunc, Review)
  for x in fit:
    data.append(x)
    return data
data['Review'] = data['Review'].apply(stopword_removal)
data.head()
Stemming
! pip install Sastrawi
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def stemming(Review):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  do = []
  for w in Review:
    dt = stemmer.stem(w)
    do.append(dt)
  d_clean=[]
  d_clean= "".join(do)
  print(d_clean)
  return d_clean
data['Review'] = data['Review'].apply(stemming)
data_clean.head()
TF -IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
text_tf = tf.fit_transform(data['Review'].astype('U'))
text_tf

Splitting Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Value'], test_size=0.2, random_state = 42)
Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:" , accuracy_score(y_test,predicted))
print("MultinomialNB Precision:" , precision_score(y_test, predicted, average="binary", pos_label="NEGATIF"))
print("MultinomialNB Recall:" , recall_score(y_test, predicted, average="binary", pos_label="NEGATIF"))
print("MultinomialNB f1_score:" , f1_score(y_test, predicted, average="binary", pos_label="NEGATIF"))

print(f'coinfusion matrix;:/n {confusion_matrix(y_test, predicted)}')
print('===============================================\n')
print(classification_report(y_test, predicted, zero_division=0))

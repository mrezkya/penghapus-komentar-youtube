import pandas as pd

# Baca data
df = pd.read_csv('output.csv')
print(df.head())

import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['komentar_bersih'] = df['Komen'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['komentar_bersih'])

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(df['Nilai'])

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediksi
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# komentar_baru = ["ambil disitus ini paling gacor", "ayo serang saja"]
# komentar_baru_bersih = [clean_text(k) for k in komentar_baru]
# X_baru = vectorizer.transform(komentar_baru_bersih)
# prediksi = model.predict(X_baru)

# for teks, label in zip(komentar_baru, encoder.inverse_transform(prediksi)):
#     print(f"'{teks}' => {label}")

def prediksi(kata):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    komentar_baru = [kata]
    komentar_baru_bersih = [clean_text(k) for k in komentar_baru]
    X_baru = vectorizer.transform(komentar_baru_bersih)
    prediksi = model.predict(X_baru)

    for teks, label in zip(komentar_baru, encoder.inverse_transform(prediksi)):
        print(f"'{teks}' => {label}")

prediksi("gacor banget")
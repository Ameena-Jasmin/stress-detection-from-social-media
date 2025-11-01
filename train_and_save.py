import pandas as pd
import nltk
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# ✅ 1. Load dataset
df = pd.read_csv("/home/ameena/Desktop/mental_health/final_dataset.csv")

# ✅ 2. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# ✅ 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.3, random_state=42)

# ✅ 4. Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ 5. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ✅ 6. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ 7. Save model and vectorizer with protocol=4 (for version safety)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f, protocol=4)

print("✅ Model and vectorizer saved successfully!")


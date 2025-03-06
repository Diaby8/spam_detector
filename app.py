import streamlit as st  # 📌 Bibliothèque pour créer l'interface web
import joblib  # 📌 Charger le modèle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 📌 Charger le modèle et le vectorizer TF-IDF
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 📌 Initialiser les objets NLP
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 📌 Fonction de nettoyage du texte
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Supprime la ponctuation
    text = re.sub(r'\s+', ' ', text)  # Supprime les espaces multiples
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# 📌 Fonction de prédiction
def predict_spam(message):
    cleaned_message = preprocess_text(message)
    vectorized_message = vectorizer.transform([cleaned_message]).toarray()
    prediction = model.predict(vectorized_message)[0]
    return prediction  # 1 = Spam, 0 = Ham

# 📌 Configuration de l'interface Streamlit
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# 📌 Ajout d'un style CSS personnalisé
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main-title {
            color: #ff4b4b;
            text-align: center;
            font-size: 40px;
        }
        .sub-title {
            text-align: center;
            font-size: 20px;
        }
        .result-box {
            padding: 15px;
            font-size: 20px;
            text-align: center;
            border-radius: 10px;
            font-weight: bold;
        }
        .spam {
            background-color: #ffcccc;
            color: #cc0000;
        }
        .ham {
            background-color: #ccffcc;
            color: #008000;
        }
    </style>
""", unsafe_allow_html=True)

# 📌 Affichage de l'interface
st.markdown('<h1 class="main-title">📩 Détecteur de Spam</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Analysez un message et détectez s\'il est SPAM ou non !</h3>', unsafe_allow_html=True)

# 📌 Zone de texte pour entrer un message
user_input = st.text_area("✍️ Tapez votre message ici :", "")

if st.button("🔍 Analyser le message"):
    if user_input:
        prediction = predict_spam(user_input)
        if prediction == 1:
            st.markdown('<div class="result-box spam">❌ SPAM DETECTÉ</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box ham">✅ HAM (Message normal)</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Veuillez entrer un message avant d'analyser.")

# 📌 Exemples de messages à tester
st.write("---")
st.write("💡 **Exemples de messages à tester :**")
examples = [
    "Congratulations! You have won a free iPhone! Click here to claim now.",
    "Hey, are we still meeting at 3 PM today?",
    "URGENT! Your bank account has been locked. Call this number now.",
    "Don't forget to bring the documents for our meeting tomorrow."
]
for ex in examples:
    st.write(f"📌 {ex}")

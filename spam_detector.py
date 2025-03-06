# Importer pandas pour manipuler les données
import pandas as pd

# Charger le dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Afficher les 5 premières lignes
print(df.head())

# Afficher les informations sur le dataset
print("\nInfos sur le dataset :")
print(df.info())

# Voir le nombre de valeurs uniques dans chaque colonne
print("\nValeurs uniques par colonne :")
print(df.nunique())

# Vérifier si on a des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Renommer les colonnes utiles
df = df.rename(columns={"v1": "label", "v2": "message"})

# Supprimer les colonnes inutiles
df = df[["label", "message"]]

# Vérifier les modifications
print("\nDataset après nettoyage :")
print(df.head())

# Afficher la répartition des classes
print("\nRépartition des messages (spam vs ham) :")
print(df["label"].value_counts())

# Visualiser avec un graphique
import matplotlib.pyplot as plt
import seaborn as sns

"""sns.countplot(x=df["label"])
plt.title("Répartition des messages (Spam vs Ham)")
plt.xlabel("Type de message")
plt.ylabel("Nombre d'occurrences")
plt.show()"""


import nltk # Bibliothèque de traitement du langage naturel (NLP)
nltk.download('stopwords')  # Liste des mots inutiles (ex: the, is, in)
nltk.download('wordnet')    # Dictionnaire pour la lemmatisation


import re #Bibliothèque pour utiliser des expressions régulières (regex) pour nettoyer le texte
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialiser le lemmatiseur et la liste des stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage du texte
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'\W', ' ', text)  # Supprimer la ponctuation
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    words = text.split()  # Découper le texte en mots
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Supprimer les stopwords et lemmatiser
    return ' '.join(words)

# Appliquer la fonction à la colonne "message"
df["cleaned_message"] = df["message"].apply(preprocess_text)

# Afficher un aperçu avant/après
print("\nAvant nettoyage :")
print(df["message"].head())

print("\nAprès nettoyage :")
print(df["cleaned_message"].head())


# Fonction de nettoyage du texte
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'\W', ' ', text)  # Supprimer la ponctuation
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    words = text.split()  # Découper le texte en mots
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Supprimer les stopwords et lemmatiser
    return ' '.join(words)

# Appliquer la fonction à la colonne "message"
df["cleaned_message"] = df["message"].apply(preprocess_text)

# Afficher un aperçu avant/après
print("\nAvant nettoyage :")
print(df["message"].head())

print("\nAprès nettoyage :")
print(df["cleaned_message"].head())

print("\nExemple de message nettoyé :")
for i in range(5):
    print(f"Original : {df['message'][i]}")
    print(f"Nettoyé  : {df['cleaned_message'][i]}")
    print("-" * 50)
    
from sklearn.feature_extraction.text import TfidfVectorizer  # Importer TF-IDF de Scikit-Learn

# Initialiser TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # On prend les 5000 mots les plus importants

# Transformer les messages en vecteurs TF-IDF
X = vectorizer.fit_transform(df["cleaned_message"]).toarray()

# Vérifier la taille des données transformées
print("\nTaille des données après vectorisation TF-IDF :", X.shape)

# Convertir les labels en valeurs numériques (ham = 0, spam = 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Vérifier les modifications
print("\nExemple de labels après conversion :")
print(df[["label", "message"]].head())



from sklearn.model_selection import train_test_split  #Importer la fonction de séparation

# Séparer les données en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, df["label"], test_size=0.2, random_state=42)

# Afficher la taille des ensembles de données
print("\nTaille des données d'entraînement :", X_train.shape)
print("Taille des données de test :", X_test.shape)


from sklearn.svm import SVC  # Importation du modèle SVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 📌 Outils pour évaluer le modèle

# Initialiser le modèle SVM
model = SVC(kernel="linear", C=1.5, class_weight="balanced", random_state=42)  # SVM avec un noyau linéaire (adapté au texte)

# Entraîner le modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Évaluer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔍 Précision du modèle (SVM) : {accuracy:.2f}")

# Afficher le rapport de classification (précision, rappel, f1-score)
print("\n📊 Rapport de classification (SVM) :")
print(classification_report(y_test, y_pred))

# Afficher la matrice de confusion
print("\n Matrice de confusion (SVM) :")
print(confusion_matrix(y_test, y_pred))

# Fonction de prédiction mise à jour avec SVM
def predict_spam(message):
    """
    Fonction qui prend un message en entrée et prédit s'il est SPAM ou HAM avec SVM.
    """
    # Nettoyer le message (mêmes étapes que précédemment)
    cleaned_message = preprocess_text(message)

    # Convertir le message en vecteur TF-IDF
    vectorized_message = vectorizer.transform([cleaned_message]).toarray()

    # Faire la prédiction avec le modèle entraîné (SVM)
    prediction = model.predict(vectorized_message)[0]

    # Retourner le résultat
    return "SPAM ❌" if prediction == 1 else "HAM ✅"

#  Tester avec des exemples de messages (inchangé)
test_messages = [
    "Congratulations! You have won a free trip to Bahamas! Click here to claim.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been suspended. Call this number now to reactivate.",
    "Don't forget to bring your laptop for the meeting.",
    "Win big now! Just send us your bank details to receive your prize."
]

#  Afficher les résultats (inchangé)
print("\n🔍 Tests du modèle SVM sur de nouveaux messages :\n")
for msg in test_messages:
    print(f"Message : {msg} \n➡️ Prédiction : {predict_spam(msg)}\n")
    print("-" * 60)



import joblib  # 📌 Bibliothèque pour sauvegarder et charger des fichiers

# 📌 Sauvegarder le modèle SVM
joblib.dump(model, "svm_spam_model.pkl")

# 📌 Sauvegarder le vectorizer TF-IDF
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Modèle SVM et vectorizer sauvegardés avec succès !")
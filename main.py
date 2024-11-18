from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Install lemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

# Read data
data = pd.read_csv('data/test.csv')
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Lemmatizing text
lemmatizer = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['intent'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
model_lr = LogisticRegression(max_iter=1000)
model_rf = RandomForestClassifier(n_estimators=100)
model_nb = MultinomialNB()

# Soft Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', model_lr), ('rf', model_rf), ('nb', model_nb)], voting='soft')
voting_clf.fit(X_train_vec, y_train)

# Predictions
predictions = voting_clf.predict(X_test_vec)

# Accuracy and Classification Report
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=voting_clf.classes_, yticklabels=voting_clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

def predict_intent(model, vectorizer):
    # Take input from the user
    input_text = input("Enter text to predict its intent: ")

    # Preprocess the text (same as training)
    input_text = input_text.lower()
    input_text = re.sub(r'[^\w\s]', '', input_text)
    input_text = ' '.join([lemmatizer.lemmatize(word) for word in input_text.split()])

    # Transform the text using the vectorizer
    input_vec = vectorizer.transform([input_text])

    # Predict the intent
    prediction = model.predict(input_vec)
    print(f"Predicted intent: {prediction[0]}")
    predict_intent(model, vectorizer)

# Call the function after training the model
predict_intent(voting_clf, vectorizer)
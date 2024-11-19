
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Install necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Read and preprocess data
data = pd.read_csv('data/test.csv')
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
lemmatizer = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['intent'], test_size=0.2, random_state=42)

# Parameters for testing
param_grid = {
    'max_features': [500, 1000],
    'n_estimators': [50, 100],
    'voting': ['soft', 'hard'],
    'weights': [(1, 1, 1), (2, 1, 1)],
}

# Results storage
results = []

# Function for preprocessing and vectorizing
def preprocess_and_vectorize(data, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(data['text'])
    return X_vec, vectorizer

# Parameter testing loop
for max_features in param_grid['max_features']:
    for n_estimators in param_grid['n_estimators']:
        for voting in param_grid['voting']:
            for weights in param_grid['weights']:
                start_time = time.time()  # Start timing

                # Vectorization
                X_train_vec, vectorizer = preprocess_and_vectorize(data, max_features)
                X_test_vec = vectorizer.transform(X_test)

                # Models
                model_lr = LogisticRegression(max_iter=500, solver='saga')
                model_et = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
                model_nb = MultinomialNB()

                # Voting Classifier
                voting_clf = VotingClassifier(
                    estimators=[('lr', model_lr), ('et', model_et), ('nb', model_nb)],
                    voting=voting,
                    weights=weights,
                    n_jobs=-1
                )

                # Training
                voting_clf.fit(X_train_vec, y_train)

                # Prediction
                predictions = voting_clf.predict(X_test_vec)

                # Evaluation
                accuracy = accuracy_score(y_test, predictions)
                elapsed_time = time.time() - start_time  # End timing

                # Save results
                results.append({
                    'max_features': max_features,
                    'n_estimators': n_estimators,
                    'voting': voting,
                    'weights': weights,
                    'accuracy': accuracy,
                    'time': elapsed_time
                })

                # Visualization
                print(f"Config: max_features={max_features}, n_estimators={n_estimators}, voting={voting}, weights={weights}")
                print(f"Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f} seconds")

                # Confusion Matrix
                cm = confusion_matrix(y_test, predictions, normalize='true')
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=voting_clf.classes_, yticklabels=voting_clf.classes_)
                plt.title(f"Confusion Matrix
(max_features={max_features}, n_estimators={n_estimators})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.show()

# Save results to DataFrame
results_df = pd.DataFrame(results)

# Show best configuration
best_result = results_df.sort_values(by='accuracy', ascending=False).iloc[0]
print("Best Configuration:", best_result)

# Save results to file
results_file = '/mnt/data/test_results_with_timing.csv'
results_df.to_csv(results_file, index=False)
print(f"Results saved to {results_file}")

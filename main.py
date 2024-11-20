import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Install necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

# Read and preprocess training and testing data
train_data = pd.read_csv('data/train.csv')

# Preprocess both datasets
train_data['text'] = train_data['text'].str.lower()
train_data['intent'] = train_data['intent'].str.lower()
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
# train_data['text'] = train_data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Your dataset
X = train_data['text']
y = train_data['intent']

# Use random_state to ensure reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# #Parameters for hard-votig with a single-model
# param_grid = {
#     'max_features': [500, 1000],
#     'n_estimators': [50, 100],  
#     'voting': ['hard'],  
#     'weights': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  
# }

# Parameters for testing 1
# param_grid = {
#     'max_features': [500, 1000],
#     'n_estimators': [50, 100],
#     'voting': ['soft', 'hard'],
#     'weights': [(1, 1, 1), (2, 1, 1)], 
# }

# Parameters for testing 2 (try increase max features)
# param_grid = {
#     'max_features': [2000],
#     'n_estimators': [50],
#     'voting': ['soft'],
#     'weights': [(1, 1, 1)],
# }

#Parameters for single model
param_grid = {
    'max_features': [1000, 2000],
    'n_estimators': [50, 100],
    'voting': ['soft'],
    'weights': [(1)],
}

# Results storage
results = []

more_visuals = True

# Parameter testing loop
for max_features in param_grid['max_features']:
    for n_estimators in param_grid['n_estimators']:
        for voting in param_grid['voting']:
            for weights in param_grid['weights']:
                start_time = time.time()  # Start timing

                # Preprocess and vectorize training and test sets separately
                vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                #Uncomment to run different models
                # # Models
                # model_lr = LogisticRegression(max_iter=500, solver='saga')
                # model_et = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
                # model_nb = MultinomialNB()

                # # Voting Classifier
                # voting_clf = VotingClassifier(
                #     estimators=[('lr', model_lr), ('et', model_et), ('nb', model_nb)],
                #     voting=voting,
                #     weights=weights,
                #     n_jobs=-1
                # )

                # # Training
                # voting_clf.fit(X_train_vec, y_train)

                # # Prediction
                # predictions = voting_clf.predict(X_test_vec)


                # Logistic Regression Model
                model_lr = LogisticRegression(max_iter=500, solver='saga')

                # Training
                model_lr.fit(X_train_vec, y_train)

                # Prediction
                predictions = model_lr.predict(X_test_vec)

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

                # Classification Report
                print(f"Config:")
                report = classification_report(y_test, predictions, target_names=train_data['intent'].unique(), output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                print(df_report.to_string(index=True))

                # Visualization
                print(f"Config: max_features={max_features}, n_estimators={n_estimators}, voting={voting}, weights={weights}")
                print(f"Accuracy: {accuracy:.4f}, Time: {elapsed_time:.2f} seconds")

                # Confusion Matrix

                #plt.show()
                plt.savefig(f"results/confusion_matrix_{max_features}_{n_estimators}_{voting}_{weights}.png")
                if(more_visuals):
                    cm = confusion_matrix(y_test, predictions, normalize='true')
                    plt.figure(figsize=(9, 7))
                    print(train_data['intent'].unique())
                    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=train_data['intent'].unique(), yticklabels=train_data['intent'].unique())
                    plt.title(f"Confusion Matrix (max_features={max_features}, n_estimators={n_estimators})")
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.savefig(f"results/confusion_matrix_{max_features}_{n_estimators}_{voting}_{weights}.png")
                    report = classification_report(y_test, predictions, target_names=train_data['intent'].unique(), output_dict=True)
                    report_df = pd.DataFrame(report).iloc[:-1, :-1]  # Exclude 'accuracy' row and 'support' column
                    plt.figure(figsize=(12, 7))
                    sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f")
                    plt.title(f"Classification Report (max_features={max_features}, n_estimators={n_estimators})")
                    plt.savefig(f"results/classification_report_{max_features}_{n_estimators}_{voting}_{weights}.png")
                    y_test_bin = label_binarize(y_test, classes=train_data['intent'].unique())

# Save results to DataFrame
results_df = pd.DataFrame(results)

# Show best configuration
best_result = results_df.sort_values(by='accuracy', ascending=False).iloc[0]
print("Best Configuration:", best_result)

# Save results to file
results_file = 'data/output.csv'

# Auto-increment logic
if os.path.exists(results_file):
    base, ext = os.path.splitext(results_file)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    results_file = f"{base}_{counter}{ext}"

results_df.to_csv(results_file, index=False)
print(f"Results saved to {results_file}")
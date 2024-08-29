import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

try:
    # Load your sentiment analysis dataset
    data = pd.read_csv('train1.csv')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train SVM model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM Model Accuracy: {accuracy}')

    # Save the model and vectorizer for later use
    with open('sentiment_model_svm.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('tfidf_vectorizer_svm.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

except FileNotFoundError:
    print("Error: Dataset file 'train1.csv' not found. Please check the file path.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

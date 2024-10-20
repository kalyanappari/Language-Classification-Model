import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import string

# Load dataset
data = pd.read_csv(r"C:\Users\asaik\Downloads\languages_dataset (1).csv")

# Preprocess text and labels
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase the text
    text = text.lower()
    return text

# Apply preprocessing to the dataset
texts = data['Texts'].apply(preprocess_text).tolist()
labels = data['Languages'].tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline that combines a CountVectorizer with a Naive Bayes classifier
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),  # Character-level n-grams
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix to check which languages are being confused
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Function to classify text
def classify_text(text):
    text = preprocess_text(text)  # Preprocess the input text
    return model.predict([text])[0]

# Chatbot interface
def chatbot():
    print("\n🎉 Welcome to the Language Classifier Bot! 🎉")
    print("I can help you identify the language of any text you provide.\n")
    
    last_prediction = None

    while True:
        # Get user input in a conversational manner
        user_input = input("🗨️ Type a sentence, phrase, or word, and I'll tell you the language: \n")

        # Classify and display the result
        predicted_language = classify_text(user_input)
        last_prediction = predicted_language
        print(f"\n🤖 Bot: The language of your text seems to be: **{predicted_language}**\n")

        # Ask if the user wants to classify another text or exit
        print("🤖 Bot: What would you like to do next?")
        print("1. Classify another text")
        print("2. See the confusion matrix")
        print("3. Repeat last classification")
        print("4. Exit")
        
        user_choice = input("Please enter the number of your choice: ").strip()
        
        if user_choice == '1':
            continue
        elif user_choice == '2':
            print("\n📊 Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print()
        elif user_choice == '3':
            if last_prediction:
                print(f"\n🔄Repeating Last Classification:\nThe language of your last text was: **{last_prediction}**\n")
            else:
                print("\n🔄 No last classification available.\n")
        elif user_choice == '4':
            print("🤖 Bot: Thanks for using the Language Classifier Bot! Goodbye! 👋")
            break
        else:
            print("🤖 Bot: I didn't quite get that. Please enter a number from 1 to 4.")
            continue

# Call the chatbot function to start interaction
chatbot()
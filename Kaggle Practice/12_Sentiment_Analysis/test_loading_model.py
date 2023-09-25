import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle


def load_sentiment_model(model_path):

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return None


def text_process(message):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Convert all text to lowercase
    4. Returns a list of the cleaned text
    5. Remove short words (e.g., words with 2 or fewer characters)
    6. Returns a list of the cleaned text
    """

    STOPWORDS = set(stopwords.words('english')).union(
        ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

    # Tokenize the text
    words = word_tokenize(message)

    # Check characters to see if they are in punctuation
    nopunc = [word for word in words if word not in string.punctuation]

    # Convert the text to lowercase and remove short words
    cleaned_words = [word.lower() for word in nopunc if len(word) > 2]

    # Now just remove any stopwords
    return ' '.join([word for word in cleaned_words if word not in STOPWORDS])


def predict_sentiment(text, model):
    try:
        # Perform the necessary text preprocessing
        # Use your text preprocessing function
        cleaned_text = text_process(text)

        # Vectorize the cleaned text
        vect = pickle.load(open("my_vect.pkl", "rb"))
        text_dtm = vect.transform([cleaned_text])

        # Predict sentiment using the loaded model
        sentiment_label = model.predict(text_dtm)[0]

        # Map the numerical sentiment label back to the original sentiment
        sentiment_mapping = {0: 'positive', 1: 'neutral', 2: 'negative'}
        predicted_sentiment = sentiment_mapping[sentiment_label]

        return predicted_sentiment
    except Exception as e:
        return None


# Example usage:
# Replace with the path to your saved model
model_path = './my_best_sentiment_model_1695566256.8833115.sav'
loaded_model = load_sentiment_model(model_path)

if loaded_model:
    input_text = "I love this product, it's amazing!"
    predicted_sentiment = predict_sentiment(input_text, loaded_model)
    print(f"Predicted Sentiment: {predicted_sentiment}")
else:
    print("Model loading failed.")

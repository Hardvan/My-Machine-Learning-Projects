import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import pandas as pd


def load_sentiment_model(model_path):

    model = joblib.load(model_path)
    return model


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


def predict_sentiment(text, model_path='./my_best_sentiment_model_1695566256.8833115.sav'):

    model = load_sentiment_model(model_path)

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


if __name__ == '__main__':

    # Define a list of input texts and their corresponding actual sentiments
    input_texts = [
        "I love this product, it's amazing!",
        "This is a neutral statement.",
        "I dislike this product, it's terrible!",
        "Great experience with the service!",
        "The movie was fantastic and thrilling.",
        "It's just okay, not too bad.",
        "This book is a masterpiece.",
        "The weather today is terrible.",
        "The event was so-so, not very exciting.",
        "I'm feeling great today."
    ]
    actual_sentiments = ['positive', 'neutral', 'negative', 'positive', 'positive',
                         'neutral', 'positive', 'negative', 'neutral', 'positive']

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(
        columns=['Input Text', 'Predicted Sentiment', 'Actual Sentiment', 'Matched'])

    # Loop through the input texts
    for input_text, actual_sentiment in zip(input_texts, actual_sentiments):
        # Predict sentiment using the loaded model
        predicted_sentiment = predict_sentiment(input_text)

        # Check if predicted and actual sentiments match
        matched = '✅' if predicted_sentiment == actual_sentiment else '❌'

        # Append the results to the DataFrame
        results_df = results_df.append({
            'Input Text': input_text,
            'Predicted Sentiment': predicted_sentiment,
            'Actual Sentiment': actual_sentiment,
            'Matched': matched
        }, ignore_index=True)

    # Display the results
    print(results_df)

import re
import nltk
from nltk.corpus import stopwords


#  NOTE: don't forget to downloasd the nltk stopwords
#  with ```nltk.download('stopwords')```
#       --> nltk.download('stopwords') <-- 
stop_words = set(stopwords.words('english'))


# inference function. It will take a movie title text and follow the below steps:
# Clean the text
# Remove stopwords from the cleaned text
# Extract features from the text
# Make predictions
# Return the predicted movie genre tags
def infer_tags(q, model, vectorizer, binarizer=None):
    # q = clean_text(q)
    # q = remove_stopwords(q)
    q_vec = vectorizer.transform([q])
    q_pred = model.predict(q_vec)

    return binarizer.inverse_transform(q_pred) if binarizer else qpred


# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


# function for text cleaning
def clean_text(text):
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()

    return text

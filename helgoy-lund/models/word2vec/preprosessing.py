from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
import settings

nltk.download('stopwords')
nltk.download('punkt')
cachedStopWords = stopwords.words("english")

# Word Preprosessing
STOPWORDING = True
STEMMING = True
FILTERING = True


def tokenize(text):
    min_length = 3

    # Tokenizing
    words = map(lambda word: word.lower(), word_tokenize(text))


    # Stopwording
    if(STOPWORDING):
        words = [word for word in words
                 if word not in cachedStopWords]


    # Stemming
    if(STEMMING):
        words = (list(map(lambda token: PorterStemmer().stem(token),
                           words)))


    p = re.compile('[a-zA-Z]+')


    # Filtering
    if(FILTERING):
        words = list(filter(lambda token:
                    p.match(token) and len(token) >= min_length,
                    words))


    return words

if __name__ == "__main__":
    print(tokenize("What does filtering do?"))
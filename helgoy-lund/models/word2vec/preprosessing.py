from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')
cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words
             if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token),
                       words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token:
                p.match(token) and len(token) >= min_length,
                tokens))
    return filtered_tokens
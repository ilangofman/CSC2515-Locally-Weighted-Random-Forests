import nltk
from load_data import load_tweet_data
import string
from num2words import num2words
from spellchecker import SpellChecker


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

def string_to_token(test_string):
    return nltk.word_tokenize(test_string)

def remove_capitals(token_str):
    for i in range(len(token_str)):
        token_str[i] = token_str[i].lower()
    return token_str

def remove_punctutation(token_str):
    new_token_str = []
    for i in range(len(token_str)):
        if token_str[i] not in string.punctuation:
            new_token_str.append(token_str[i])
    return new_token_str


def remove_stopwords(token_str):
    stop_words = nltk.corpus.stopwords.words('english')
    new_token_str = []
    for i in range(len(token_str)):
        if token_str[i] not in stop_words:
            new_token_str.append(token_str[i])
    return new_token_str

def correct_spelling(token_str):
    spell = SpellChecker()
    for i in range(len(token_str)):
        token_str[i] = spell.correction(token_str[i])
    return token_str

def translate_numbers(token_str):
    for i in range(len(token_str)):
        if token_str[i].isnumeric():
            token_str[i] = num2words(token_str[i])
    return token_str

def lematize(test_string):
    pass

def token_to_string(token_str):
    return " ".join(token_str)

def clean_string(test_string):
    token_str = string_to_token(test_string)
    token_str = remove_capitals(token_str)
    token_str = remove_punctutation(token_str)
    token_str = remove_stopwords(token_str)
    token_str = correct_spelling(token_str)
    token_str = translate_numbers(token_str)

    test_string = token_to_string(token_str)
    

    return test_string

print(clean_string("The 23 kuick and brown Fox, jumps over the lazy Dog."))
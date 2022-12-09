from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from spellchecker import SpellChecker

from load_data import load_tweet_data

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except:
    nltk.download('omw-1.4')

def string_to_token(test_string):
    return nltk.word_tokenize(test_string)

def remove_capitals(token_str):
    return [tok.lower() for tok in token_str]

def remove_punct_stop(token_str):
    return [tok for tok in token_str if \
        (tok not in stopwords.words('english')) and (tok not in punctuation)]

def correct_spelling(token_str):
    spell = SpellChecker()
    return [spell.correction(tok) for tok in token_str]

def translate_numbers(token_str):
    return [num2words(tok) if tok.isnumeric() else tok for tok in token_str]

def num_toeknize(token_str):
    return ["NUMBER" if tok.isnumeric() else tok for tok in token_str]

def lemmatize(token_str):
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(tok) if tok!="NUMBER" else "NUMBER" for tok in token_str]

def token_to_string(token_str):
    return " ".join(token_str)

def clean_string(test_string):
    token_str = string_to_token(test_string)
    token_str = remove_capitals(token_str)
    token_str = remove_punct_stop(token_str)
    # token_str = correct_spelling(token_str)
    # token_str = translate_numbers(token_str)
    token_str = num_toeknize(token_str)
    token_str = lemmatize(token_str)
    test_string = token_to_string(token_str)
    

    return test_string
if __name__ == "__main__":
    print(clean_string("The 23 quick and brown Fox, jumps over the lazy Dog testing some verbs running sprint tasted saw."))
# Import libraries
import re
import pandas as pd
from collections import Counter

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet


### NORMALIZATION #############################################################

### FUNCTIONS APPLIED TO TEXT CORPUS ###

# FUNCTION: Merge hyphenated words at the end of line & remove line breaks
def line_breaks(text):
    
    # find combination of hypened words with double line spacing
    matchList = re.findall(r"\w+-\n\n", text)
    proc = text
    for m in matchList:
        new = m.replace("-\n\n","")
        proc = proc.replace(m,new)

    # find combination of hyphened words with single line spacing
    matchList = re.findall(r"\w+-\n", proc)
    for m in matchList:
        new = m.replace("-\n","")
        proc = proc.replace(m,new)

    # remove line breaks
    matchList = re.findall(r"\n", proc)
    out = proc
    for m in matchList:
        new = m.replace("\n"," ")
        out = out.replace(m,new)

    return out


# FUNCTION: Remove URLs
def remove_urls(text):

    url = r"\S+www\S+"
    text = re.sub(url, "", text)
    
    return text


# FUNCTION: Remove parenthesised text
def remove_parentheses(text):

    text = re.sub(r'\([^)]*\)', "", text)
    
    return text


# FUNCTION: Replace abbreviations by their long form 
def replace_terms(text, dictionary):
   
    for item in dictionary.keys():
        abbrev = r"\b{}\b".format(item)
        text = re.sub(abbrev, dictionary[item], text)
   
    return text



### FUNCTIONS APPLIED TO TOKENS (SENT & WORD) ###

# FUNCTION: normalize notation
def norm_notation(sent, dictionary):
    
    words = []
    for word in sent:
        if word in dictionary:
            word = dictionary[word]
        
        words.append(word)
   
    return words


# FUNCTION: lowercase
def to_lower(sent):

    words = []
    for word in sent:
        word = word.lower()
        words.append(word)

    return words


# FUNCTION: remove non-alphabetic characters
def rm_nonalpha(sent):
    
    words = []
    for word in sent:
        if word.isalpha():
            words.append(word)

    return words


# FUNCTION: remove stopwords
def rm_stopword(sent, custom_stopwords):

    words = []
    for word in sent:
        if not (word in custom_stopwords):
            words.append(word)

    return words


# FUNCTION: remove single-character words
def min_len(sent):

    words = []
    for word in sent:
        if len(word)>1:
            words.append(word)

    return words


# FUNCTION: replace rare words by dummy-word
def replace_rare(sent, dictionary):

    exceptions = list(dictionary.keys())

    text = []
    for sentence in sent:
        for word in sentence:
            text.append(word)
    counts = dict(Counter(text))

    rare = []
    for key in counts.keys():
        if (counts[key] < 50) and (key not in exceptions):
            rare.append(key)

    for sentence in sent:
        for i in range(0, len(sentence)):
            if sentence[i] in rare:
                sentence[i] = "RARE"

    return sent


###############################################################################


# TOKENIZATION ################################################################

# FUNCTION: split text into sentences 
def text_into_sents(text):
    return sent_tokenize(text)


# FUNCTION: split sentences into words
def sent_into_words(sents):
    
    sentences = []

    for sent in sents:
        word_tokens = word_tokenize(sent)
        sentences.append(word_tokens)

    return sentences


###############################################################################


# MISCELLANEOUS ###############################################################

# FUNCTION: POS tagging
def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
# Import libraries
import re


# FUNCTION: handling of n-grams
def ngrams(corpus, dictionary):
    corpusNew = []

    for sent in corpus:
        fullText = " ".join(sent)
        sentence = []

        i = 0
        while(i < len(dictionary.keys())):
            item = list(dictionary.keys())[i]
            fullText = re.sub(item, dictionary[item], fullText)
            i += 1

        words = fullText.split()
        for word in words:
            sentence.append(word)

        corpusNew.append(sentence)

    return corpusNew
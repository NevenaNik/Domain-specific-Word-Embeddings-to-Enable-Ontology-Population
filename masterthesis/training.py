# TODO
"""
--> add paramters to customize Word2vec model

"""

import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
import glob
import pickle
import random

from gensim.models import Phrases
from gensim.models import Word2Vec

from config import *



def get_options():
    parser = argparse.ArgumentParser(description="Preprocessing raw txt files")

    parser.add_argument("-r", "--reduction",
                        help="Reduction type of data to train model on (default: lemmatization)",
                        choices=["none", "stem", "lemma"],
                        default="lemma")
    parser.add_argument("-s", "--source",
                        help="Source directory with files for model training",
                        default=path_data_proc)
    parser.add_argument("-d", "--destination",
                        help="Destination directory for trained model",
                        default=path_models)
    parser.add_argument("-n", "--ngrams",
                        help="Consider compound terms (default: up to trigrams)",
                        choices=["unigram", "bigram", "trigram"],
                        default="trigram")
    parser.add_argument("--shuffle",
                        help="shuffling of sentences before training",
                        action="store_true")                    
    parser.add_argument("-l", "--logging",
                        help="Set logging level (optional)",
                        choices=["INFO", "DEBUG", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    return args







class Model(object):

    def __init__(self, **kwargs):

        self.logger = logging.getLogger("word2vec")
        if kwargs.get("logging", None):
            self.logger.setLevel(kwargs.get("logging"))


        # Get data
        self.source = kwargs.get("source", path_data_proc)

        self.reduction = kwargs.get("reduction", "lemma")
        if self.reduction == "lemma":
            self.files = glob.glob(self.source + "*_<WordNetLemmatizer>.pickle")
        elif self.reduction == "stem":
            self.files = glob.glob(self.source + "*_<PorterStemmer>.pickle")
        else:
            self.files = glob.glob(self.source + "*_None.pickle")

        self.logger.info(f"Reduction method in preprocessing: {self.reduction}")

        # Consideration of compound terms
        self.ngram = kwargs.get("ngrams", "trigram")
        self.logger.info(f"Compound terms up to: {self.ngram}")

        # Shuffling of sentences
        self.shuffle = kwargs.get("shuffle")
        self.logger.info(f"Shuffling: {self.shuffle}")
        
        # Set destination directory
        self.dest = kwargs.get("destination", path_models)



    def execute(self):

        corpus = []

        for filename in self.files:
            corpus = corpus + pickle.load(open(filename, "rb"))

        if self.shuffle:
            random.seed(42)
            random.shuffle(corpus)
        
        sentences = corpus
        if self.ngram != "unigram":
            bigram = Phrases(sentences)
            sentences = list(bigram[sentences])
            if self.ngram == "trigram":
                trigram = Phrases(sentences)
                sentences = list(trigram[sentences])

       
        model = Word2Vec(sentences)

        if self.shuffle:
            file_name = "word2vec_" + str(self.reduction) + "_" + str(self.ngram) + "_shuffled.bin"
        else:
            file_name = "word2vec_" + str(self.reduction) + "_" + str(self.ngram) + ".bin"
        model.save(self.dest + file_name)

        self.logger.info(f"Model training done. Model saved as: {file_name}")
        



if __name__ == '__main__':
    options = get_options()
    model = Model(**vars(options))
    model.execute()

    sys.exit(0)
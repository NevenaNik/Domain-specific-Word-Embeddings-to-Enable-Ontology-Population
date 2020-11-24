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
import json
import random

from gensim.models import Phrases
from gensim.models import Word2Vec

from functions.fun_training import *
from functions.config import *



def get_options():
    parser = argparse.ArgumentParser(description="Preprocessing raw txt files")

    parser.add_argument("-r", "--reduction",
                        help="Reduction type of data to train model on (default: lemmatization)",
                        choices=["none", "stem", "lemma"],
                        default="lemma")
    parser.add_argument("--rare",
                        help="replacement of rare words with placeholder",
                        action="store_true")
    parser.add_argument("-s", "--source",
                        help="Source directory with files for model training",
                        default=path_preprocessed)
    parser.add_argument("-d", "--destination",
                        help="Destination directory for trained model",
                        default=path_trained)
    parser.add_argument("--ngramsMan",
                        help="Manual induction of n-grams (default: during trainig)",
                        action="store_true")
    parser.add_argument("--shuffle",
                        help="Shuffling of sentences before training",
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
        self.source = kwargs.get("source", path_preprocessed)

        self.reduction = kwargs.get("reduction", "lemma")
        self.rare = kwargs.get("rare")
        if self.reduction == "lemma":
            if self.rare:
                self.files = glob.glob(self.source + "*_<WordNetLemmatizer>_rareTrue.pickle")
            else:
                self.files = glob.glob(self.source + "*_<WordNetLemmatizer>_rareFalse.pickle")
        elif self.reduction == "stem":
            if self.rare:
                self.files = glob.glob(self.source + "*_<PorterStemmer>_rareTrue.pickle")
            else:
                self.files = glob.glob(self.source + "*_<PorterStemmer>_rareFalse.pickle")
        else:
            if self.rare:
                self.files = glob.glob(self.source + "*_None_rareTrue.pickle")
            else:
                self.files = glob.glob(self.source + "*_None_rareFalse.pickle")

        self.logger.info(f"Reduction method in preprocessing: {self.reduction}")
        self.logger.info(f"Handling of rare words: {self.rare}")

        # Handling of n-grams
        self.ngrams = kwargs.get("ngramsMan")
        if self.ngrams:
            self.ng = "Manual"
            self.logger.info("Handling of n-grams: manual")
        else:
            self.ng = "Training"
            self.logger.info("Handling of n-grams: in training")

        # Shuffling of sentences
        self.shuffle = kwargs.get("shuffle")
        self.logger.info(f"Shuffling: {self.shuffle}")
        
        # Set destination directory
        self.dest = kwargs.get("destination", path_trained)



    def execute(self):

        corpus = []

        for filename in self.files:
            corpus = corpus + pickle.load(open(filename, "rb"))

        # Shuffling
        if self.shuffle:
            random.seed(42)
            random.shuffle(corpus)
        
        # Handling of n-grams
        if self.ngrams:
            dictionary = json.loads(open(path_ngrams).read())
            sentences = ngrams(corpus, dictionary)
        else:
            bigram = Phrases(corpus)
            sentences = list(bigram[corpus])
            
            trigram = Phrases(sentences)
            sentences = list(trigram[sentences])

        # Model training
        model = Word2Vec(sentences)

        # Save model
        file_name = "word2vec_" + str(self.reduction) + "_rare" + str(self.rare) + "_ngrams" + str(self.ng) + "_shuffled" + str(self.shuffle) + ".bin"
        model.save(self.dest + file_name)

        self.logger.info(f"Model training done. Model saved as: {file_name}")
        



if __name__ == '__main__':
    options = get_options()
    model = Model(**vars(options))
    model.execute()

    sys.exit(0)
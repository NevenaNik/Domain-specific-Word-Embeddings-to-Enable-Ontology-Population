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

from functions.fun_training import ngrams
from functions.config import path_preprocessed, path_models, path_ngrams



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
                        default=path_models)
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

    parser.add_argument("-vs", "--vector_size",
                        help="dimensionality of the word vectors",
                        default=100, type=int)
    parser.add_argument("-win", "--window_size",
                        help="max distance between current and the predicted word within sentence",
                        default=5, type=int)
    parser.add_argument("-mc", "--min_count",
                        help="ignores all words with a total freq lower than this",
                        default=5, type=int)
    parser.add_argument("--skipgram",
                        help="training algorithm: skipgram (default: CBOW)",
                        action="store_true")
    parser.add_argument("--negSampl",
                        help="Use negative sampling for model training (default: hierarchical softmax)",
                        action="store_true")

    parser.add_argument("-p", "--percentage",
                        help="percentage of data to be used for training (default: 1 = 100 percent)",
                        default=1, type=float)
    args = parser.parse_args()
    return args







class Model(object):

    def __init__(self, **kwargs):

        self.logger = logging.getLogger("word2vec")
        if kwargs.get("logging", None):
            self.logger.setLevel(kwargs.get("logging"))

        self.fh = logging.FileHandler("train.log")
        self.fh.setLevel(kwargs.get("logging"))
        self.logger.addHandler(self.fh)


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


        # Model param: vector size
        self.vs = kwargs.get("vector_size", 100)
        self.logger.info(f"Vector size: {self.vs}")

        # Model param: window size
        self.win = kwargs.get("window_size", 5)
        self.logger.info(f"Window size: {self.win}")

        # Model param: min count
        self.mc = kwargs.get("min_count", 5)
        self.logger.info(f"Min count: {self.mc}")

        # Model param: training algorithm
        self.train = kwargs.get("skipgram")
        if self.train:
            self.sg = 1
            self.logger.info("Train algorithm: skipgram")
        else:
            self.sg = 0
            self.logger.info("Train algorithm: CBOW")

        # Model param: hierarchical softmax vs. negative sampling
        self.sampl = kwargs.get("negSampl")
        if self.sampl:
            self.hs = 0
            self.logger.info("used in training: negative sampling")
        else:
            self.hs = 1
            self.logger.info("used in training: hierarchical softmax")

        
        # Training data - percentage of data to be used
        self.percentage = kwargs.get("percentage", 1)
        self.logger.info(f"Percentage of training data used: {self.percentage * 100}%")
        
        # Set destination directory
        self.dest = kwargs.get("destination", path_models)



    def execute(self):

        corpus = []

        for filename in self.files:
            corpus = corpus + pickle.load(open(filename, "rb"))

        # Shuffling
        if self.shuffle:
            random.seed(42)
            random.shuffle(corpus)

        # Sampling Training data (only consider xx% of all data)
        random.seed(42)
        size = int(round(self.percentage * len(corpus)))
        corpus = random.sample(corpus, size)
        
        # Handling of n-grams
        if self.ngrams:
            dictionary = json.loads(open(path_ngrams).read())
            sentences = ngrams(corpus, dictionary)
        else:
            bigram = Phrases(corpus)
            sentences = list(bigram[corpus])
            
            trigram = Phrases(sentences)
            sentences = list(trigram[sentences])

        # Save training data corpus (for stats)
        pathCorpus = "../data/corpora/"
        nameCorpus = "corpus_percent" + str(int(self.percentage*100)) + ".txt"
        fileCorpus = open(pathCorpus+nameCorpus, "a")

        for sent in sentences:
            newLine = " ".join(sent)
            newLine = newLine + "\n"
            fileCorpus.write(newLine)
        
        fileCorpus.close()

        # Model training
        model = Word2Vec(sentences, size=self.vs, window=self.win, min_count=self.mc, sg=self.sg, hs=self.hs)

        # Save model
        file_name = f"w2v_{self.reduction}_rare{self.rare}_ngrams{self.ng}_shuffled{self.shuffle}_vs{self.vs}_win{self.win}_mc{self.mc}_sg{self.sg}_hs{self.hs}_size{self.percentage}.bin"
        model.save(self.dest + file_name)

        self.logger.info(f"Model training done. Model saved as: {file_name}")
        self.logger.info("----------------------------------------------------------")
        



if __name__ == '__main__':
    options = get_options()
    model = Model(**vars(options))
    model.execute()

    sys.exit(0)
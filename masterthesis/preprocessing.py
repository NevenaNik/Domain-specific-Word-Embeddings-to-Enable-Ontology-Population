import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys

import glob
import json
import pickle

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from functions.fun_preprocessing import *
from functions.config import *
from reduction import WordNetLemmatizer, PorterStemmer





def get_options():
    parser = argparse.ArgumentParser(description="Preprocessing raw txt files")

    parser.add_argument("-f", "--file",
                        help="Txt file to be preprocessed")
    parser.add_argument("-s", "--source",
                        help="Source directory with txt files to be preprocessed",
                        default=path_raw)
    parser.add_argument("-d", "--destination",
                        help="Destination directory for processed files",
                        default=path_preprocessed)
    parser.add_argument("-r", "--reduction",
                        help="Choose form of reduction (default: lemmatization)",
                        choices=["none", "stem", "lemma"],
                        default="lemma")
    parser.add_argument("--rare",
                        help="replacement of rare words with placeholder",
                        action="store_true")                    
    parser.add_argument("-l", "--logging",
                        help="Set logging level (optional)",
                        choices=["INFO", "DEBUG", "ERROR"],
                        default="INFO")
    args = parser.parse_args()
    return args



class Preprocess(object):

  def __init__(self, **kwargs):

    self.logger = logging.getLogger("preprocessing")
    if kwargs.get("logging", None):
      self.logger.setLevel(kwargs.get("logging"))


    # Get files 
    self.source = kwargs.get("source", path_raw)

    self.__file = kwargs.get("file", None)
    if self.__file is None:
      self.files = glob.glob(self.source + "*")
    else:
      self.files = glob.glob(self.source + self.__file)

    self.logger.info(f"Number of files to be processed: {len(self.files)}")

    # Get form of reduction
    reduction = kwargs.get("reduction", "lemma")
    if reduction == "lemma":
      self.reduction_method = WordNetLemmatizer()
    elif reduction == "stem":
      self.reduction_method = PorterStemmer()  
    else:
      self.reduction_method = None

    self.logger.info(f"Reduction method: {self.reduction_method}")

    # Handling of rare words
    self.rare = kwargs.get("rare")
    self.logger.info(f"Handling of rare words: {self.rare}")

    # Set destination directory
    self.dest = kwargs.get("destination", path_preprocessed)

    


  def execute(self):
    
    term_abbrev = json.loads(open(path_abbreviations).read())
    term_normal = json.loads(open(path_normalization).read())
    term_ngrams = json.loads(open(path_ngrams).read())
    
    custom_stopwords = json.loads(open(path_customStopwords).read())
    custom_stopwords = set(stopwords.words('english') + custom_stopwords["stopwords"])

    for i in range(1, len(self.files)+1):
      source = self.files[i-1]
      raw = open(source, 'r').read()

      self.logger.info(f"Processing of file {i}/{len(self.files)}: {self.files[i-1]}")

      proc = raw
      proc = line_breaks(proc)
      proc = remove_urls(proc)
      proc = remove_parentheses(proc)
      proc = replace_terms(proc, term_abbrev)
      proc = replace_terms(proc, term_normal)
            
      proc = text_into_sents(proc)
      proc = sent_into_words(proc)

      sentences = []
      for sent in proc:
        sent = to_lower(sent)
        sent = rm_nonalpha(sent)
        sent = rm_stopword(sent, custom_stopwords)
        sent = min_len(sent)
        if len(sent)!=0:
          sentences.append(sent)
      
      reduced = sentences
      if self.reduction_method != None:
        reduced = []
        for sent in sentences:
          sent = " ".join(sent)
          sent = [self.reduction_method.dim_reduction(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(sent)]
          reduced.append(sent)

      if self.rare:
        reduced = replace_rare(reduced, term_ngrams)
      
      file_name = "preproc_" + str(i) + "_" + str(self.reduction_method) + "_rare" + str(self.rare) + ".pickle"
      with open(self.dest + file_name, "wb") as fp:
        pickle.dump(reduced, fp)
      
      self.logger.info(f"Finished processing {self.files[i-1]} - saved as: {file_name}")
  



if __name__ == '__main__':
    options = get_options()
    preproc = Preprocess(**vars(options))
    preproc.execute()

    sys.exit(0)
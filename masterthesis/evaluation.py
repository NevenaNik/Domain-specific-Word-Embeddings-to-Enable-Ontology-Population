import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
import json
import pickle
import numpy as np

from gensim.models import Word2Vec
from collections import Counter
from itertools import repeat
from scipy.stats import spearmanr
from voting import vote_aggreg

from config import *


def get_options():
    parser = argparse.ArgumentParser(description="Intrinsic evaluation")

    parser.add_argument("-m", "--model",
                        help="Model to be evaluated",
                        required=True)
    parser.add_argument("-s", "--source",
                        help="Source directory with model file",
                        default=path_models)
    parser.add_argument("-r", "--results",
                        help="Filename for evaluation results")                
    parser.add_argument("-d", "--destination",
                        help="Destination directory for evaluation results",
                        default=path_data_eval)
    parser.add_argument("-v", "--voting",
                        help="Type of voting: weighted, majority (weighted), majority (unweighted)",
                        choices=["weighted", "majority_w", "majority_uw"],
                        default="weighted")
    parser.add_argument("-l", "--logging",
                        help="Set logging level (optional)",
                        choices=["INFO", "DEBUG", "ERROR"],
                        default="INFO")
    
    args = parser.parse_args()
    return args




class Intrinsic(object):

    def __init__(self, **kwargs):

        self.logger = logging.getLogger("evaluation")
        if kwargs.get("logging", None):
            self.logger.setLevel(kwargs.get("logging"))


        # Get model
        self.source = kwargs.get("source", path_models)
            
        self._file = kwargs.get("model", None)
        if self._file is None:
            raise TypeError()

        self.model = str(self.source) + str(self._file)

            
        # Set type of vote aggregation
        self.vote = kwargs.get("voting", "weigthed")
            
        # Set results file
        self.dest = kwargs.get("destination", path_data_eval)

        self._results = kwargs.get("results", None)
        if self._results is None:
            self.results = None
        else:
            self.results = str(self.dest) + str(self._results)

        


    def execute(self):

        # Load vocabulary list
        intrinsic_eval = json.loads(open(path_vocab_lists+"intrinsic_eval_key.json").read())
        intrinsic_eval = intrinsic_eval["eval_list"]

        # Expert voting
        expert = {}

        for key in intrinsic_eval.keys():
            filename = intrinsic_eval[key]
            votes = json.loads(open(path_expert+filename).read())
            expert[key] = vote_aggreg(self.vote, votes)

        # Load model
        model = Word2Vec.load(self.model)

        # Results
        vote_expert = []
        vote_model = []

        for key in expert.keys():
            for term in expert[key].keys():
                try:
                    score = model.wv.similarity(key, term)
                    vote_model.append(score)
                    vote_expert.append(expert[key][term])
                except:
                    print(f"No similarity score for {key} and {term}.")
                    pass

        print(len(vote_expert))
        print(len(vote_model))
        print(spearmanr(vote_expert, vote_model))
        
            

if __name__ == '__main__':
    options = get_options()
    evaluate = Intrinsic(**vars(options))
    evaluate.execute()
        
    sys.exit(0)
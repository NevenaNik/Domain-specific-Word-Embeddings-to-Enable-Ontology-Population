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

from functions.fun_evaluation import vote_aggreg, accuracy
from functions.config import *


def get_options():
    parser = argparse.ArgumentParser(description="Intrinsic evaluation")

    parser.add_argument("-m", "--model",
                        help="Model to be evaluated",
                        required=True)
    parser.add_argument("-s", "--source",
                        help="Source directory with model file",
                        default=path_trained)
    parser.add_argument("-d", "--destination",
                        help="Destination directory for evaluation results",
                        default=path_eval)
    parser.add_argument("-a", "--aggregation",
                        help="Aggregation of expert opinion: weighted average, weighted majority vote, unweighted majority vote",
                        choices=["weighted", "majorityWeighted", "majorityUnweighted"],
                        default="weighted")
    parser.add_argument("--devCheck",
                        help="Only consider data points with consistent expert raiting",
                        action="store_true")
    parser.add_argument("--accuracy",
                        help="Evaluate model by accuracy score (default: spearman correlation",
                        action="store_true")
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
        self.source = kwargs.get("source", path_trained)
            
        self._file = kwargs.get("model", None)
        if self._file is None:
            raise TypeError()

        self.model = str(self.source) + str(self._file)

            
        # Set type of aggregation (expert rating)
        self.aggreg = kwargs.get("aggregation", "weigthed")

        # Check for consistency (expert rating)
        self.devCheck = kwargs.get("devCheck")

        # Set evaluation method
        self.eval = kwargs.get("accuracy")
        if self.eval:
            self.logger.info("Model evaluation by: accuracy score")
        else:
            self.logger.info("Model evaluation by: spearman correlation")


        # Set results file
        self.dest = kwargs.get("destination", path_eval)

        self._results = kwargs.get("results", None)
        if self._results is None:
            self.results = str(self.dest) + "eval_results.txt"
        else:
            self.results = str(self.dest) + str(self._results)

        


    def execute(self):

        # Load vocabulary list
        intrinsic_eval = json.loads(open(path_evalIntrinsic).read())
        intrinsic_eval = intrinsic_eval["eval_list"]

        # Expert voting
        expert = {}

        for key in intrinsic_eval.keys():
            filename = intrinsic_eval[key]
            votes = json.loads(open(path_expert+filename).read())
            expert[key] = vote_aggreg(self.aggreg, votes)

        # Load model
        model = Word2Vec.load(self.model)

        # Results
        vote_expert = []
        vote_model = []

        for key in expert.keys():
            for term in expert[key].keys():
                if self.devCheck:
                    if expert[key][term][1] < 0.5:
                        try:
                            score = model.wv.similarity(key, term)
                            vote_model.append(score)
                            vote_expert.append(expert[key][term])
                        except:
                            print(f"No similarity score for {key}/{term}.")
                            pass
                    else:
                        print(f"Deviation of expert rating for {key}/{term} too large.")
                else:
                    try:
                        score = model.wv.similarity(key, term)
                        vote_model.append(score)
                        vote_expert.append(expert[key][term])
                    except:
                        print(f"No similarity score for {key}/{term}.")
                        pass

        print(len(vote_expert))
        print(len(vote_model))

        if self.aggreg:
            score = accuracy(vote_expert, vote_model)
        else:
            score = spearmanr(vote_expert, vote_model)
        print(score)
        
            

if __name__ == '__main__':
    options = get_options()
    evaluate = Intrinsic(**vars(options))
    evaluate.execute()
        
    sys.exit(0)
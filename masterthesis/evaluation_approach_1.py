import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import sys
import json
import pickle
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from collections import Counter
from itertools import repeat
from scipy.stats import spearmanr

from functions.fun_evaluation import vote_aggreg, accuracy
from functions.config import path_results, path_expert, path_models, path_eval1


def get_options():
    parser = argparse.ArgumentParser(description="Evaluation approach 1")

    parser.add_argument("--id",
                        help="id to match log files")
    parser.add_argument("-m", "--model",
                        help="Model to be evaluated",
                        required=True)
    parser.add_argument("-s", "--source",
                        help="Source directory with model file",
                        default=path_models)
    """
    parser.add_argument("-d", "--destination",
                        help="Destination directory for evaluation results",
                        default=path_eval)
    """                  
    parser.add_argument("-a", "--aggregation",
                        help="Aggregation of expert opinion: weighted average (default), weighted majority vote, unweighted majority vote",
                        choices=["weighted", "majorityWeighted", "majorityUnweighted"],
                        default="weighted")
    parser.add_argument("--devCheck",
                        help="Only consider data points with consistent expert raiting",
                        action="store_true")
    parser.add_argument("--spearman",
                        help="Evaluate model by spearman correlation (default: accuracy score)",
                        action="store_true")
    parser.add_argument("-e", "--evalList",
                        help="Provide path to evaluation list for consistency (default: all word pairs will be used)")
    parser.add_argument("-l", "--logging",
                        help="Set logging level (optional)",
                        choices=["INFO", "DEBUG", "ERROR"],
                        default="INFO")
    
    args = parser.parse_args()
    return args




class Approach1(object):

    def __init__(self, **kwargs):

        self.logger = logging.getLogger("evaluation")
        if kwargs.get("logging", None):
            self.logger.setLevel(kwargs.get("logging"))
        
        self.fh = logging.FileHandler("eval.log")
        self.fh.setLevel(kwargs.get("logging"))
        self.logger.addHandler(self.fh)

        self.id = kwargs.get("id", None)

        # Get model
        self.source = kwargs.get("source", path_models)
            
        self._file = kwargs.get("model", None)
        if self._file is None:
            raise TypeError()
        self.logger.info(f"Evaluating model: {self._file}")
        self.model = str(self.source) + str(self._file)

            
        # Set type of aggregation (expert rating)
        self.aggreg = kwargs.get("aggregation", "weigthed")
        self.logger.info(f"Expert rating aggregation: {self.aggreg}")

        # Check for consistency (expert rating)
        self.devCheck = kwargs.get("devCheck")
        self.logger.info(f"Expert rating consistensy check: {self.devCheck}")

        # Set evaluation method
        self.eval = kwargs.get("spearman")
        if self.eval:
            self.logger.info("Model evaluation by: spearman correlation")
        else:
            self.logger.info("Model evaluation by: accuracy score")

        # Get eval list
        self.evalList = kwargs.get("evalList", None)

        # Set results file
        self.dest = kwargs.get("destination", path_results)

        self._results = kwargs.get("results", None)
        if self._results is None:
            self.results = str(self.dest) + "eval_results.txt"
        else:
            self.results = str(self.dest) + str(self._results)

        


    def execute(self):

        # Open result log
        pathEval = "evalLog.csv"
        evalLog = open(pathEval, "a")
        newline = []
        newline.append(str(self.id))
        newline.append(str(self.aggreg))
        newline.append(str(self.devCheck))
        newline.append("accuracy score")

        # Load vocabulary list
        approach1 = json.loads(open(path_eval1).read())
        approach1 = approach1["eval_list"]

        # Expert voting
        expert = {}

        for key in approach1.keys():
            filename = approach1[key]
            votes = json.loads(open(path_expert+filename).read())
            expert[key] = vote_aggreg(self.aggreg, votes)

        # Load model
        model = Word2Vec.load(self.model)

      
        # Results
        vote_expert = []
        vote_model = []

        if self.evalList != None: # use consistent eval list
            
            pathPairs = self.evalList
            evalPairs = pd.read_csv(pathPairs, sep=";", names=["key", "term"])

            pairs = []
            for index, row in evalPairs.iterrows():
                pair = [row["key"], row["term"]]
                pairs.append(pair)

            for key in expert.keys():
                for term in expert[key].keys():
                    if self.devCheck:
                        if (expert[key][term][1] < 1.25) & ([str(key), str(term)] in pairs):
                            try:
                                score = model.wv.similarity(key, term)
                                vote_model.append(score)
                                vote_expert.append(expert[key][term][0])
                            except:
                                #print(f"No similarity score for {key}/{term}.")
                                pass
                        else:
                            #print(f"Deviation of expert rating for {key}/{term} too large.")
                            pass
                    else:
                        try:
                            score = model.wv.similarity(key, term)
                            vote_model.append(score)
                            vote_expert.append(expert[key][term][0])
                        except:
                            #print(f"No similarity score for {key}/{term}.")
                            pass

        else: # evaluate with all word pairs (number of pairs may not be consistent per model)
            
            for key in expert.keys():
                for term in expert[key].keys():
                    if self.devCheck:
                        if expert[key][term][1] < 1.25:
                            try:
                                score = model.wv.similarity(key, term)
                                vote_model.append(score)
                                vote_expert.append(expert[key][term][0])
                            except:
                                #print(f"No similarity score for {key}/{term}.")
                                pass
                        else:
                            #print(f"Deviation of expert rating for {key}/{term} too large.")
                            pass
                    else:
                        try:
                            score = model.wv.similarity(key, term)
                            vote_model.append(score)
                            vote_expert.append(expert[key][term][0])
                        except:
                            #print(f"No similarity score for {key}/{term}.")
                            pass

       
        if self.eval:
            score = spearmanr(vote_expert, vote_model)
        else:
            score = accuracy(vote_expert, vote_model)
        
        self.logger.info(f"Count of word pairs: {len(vote_expert)}")
        self.logger.info(f"Evaluation score: {score}")
        self.logger.info("----------------------------------------")  

        newline.append(str(len(vote_expert)))  
        newline.append(str(score))
        newline = ";".join(newline)
        newline = newline + "\n"
        evalLog.write(newline)
        evalLog.close()
            

if __name__ == '__main__':
    options = get_options()
    evaluate = Approach1(**vars(options))
    evaluate.execute()
        
    sys.exit(0)
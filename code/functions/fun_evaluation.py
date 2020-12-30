# Import libraries
import statistics
from collections import Counter
from sklearn.metrics import accuracy_score




# FUNCTION: vote aggregation by...
#                       ...(1) weighted average
#                       ...(2) majority vote (weighted)
#                       ...(3) majority vote (unweighted)
def vote_aggreg(method, dictionary):
    
    votes = {}

    focus_term = list(dictionary.keys())[0]
    dictionary = dictionary[focus_term]

    if method == "weighted":
        for term in dictionary.keys():
            vote = []
            for item in dictionary[term]:
                vote.extend([item[0]] * (5-item[1]))
            mean = statistics.mean(vote)
            try:
                stdev = statistics.stdev(vote)
            except:
                stdev = 0
            votes[term] = [mean, stdev]

    elif method == "majorityWeighted":
        for term in dictionary.keys():
            vote = []
            for item in dictionary[term]:
                vote.extend([item[0]] * (5-item[1]))

            count = Counter(vote)
            mode = []
            mode.append(count.most_common()[0][0])
            i = 1
            cond = (len(count)>i)
            while(cond):
                if (count.most_common()[0][1] == count.most_common()[i][1]):
                    mode.append(count.most_common()[i][0])
                    i += 1
                    cond = (len(count)>i)
                else:
                    cond = False
            mean = statistics.mean(mode)
            try:
                stdev = statistics.stdev(vote)
            except:
                stdev = 0
            votes[term] = [mean, stdev]            
    else:
        for term in dictionary.keys():
            vote = []
            for item in dictionary[term]:
                vote.extend([item[0]])

            count = Counter(vote)
            mode = []
            mode.append(count.most_common()[0][0])
            i = 1
            cond = (len(count)>i)
            while(cond):
                if (count.most_common()[0][1] == count.most_common()[i][1]):
                    mode.append(count.most_common()[i][0])
                    i += 1
                    cond = (len(count)>i)
                else:
                    cond = False
            mean = statistics.mean(mode)
            try:
                stdev = statistics.stdev(vote)
            except:
                stdev = 0
            votes[term] = [mean, stdev]

    return votes


# FUNCTION: Model accuracy (based on expert ratings)
def accuracy(expert, model):
    if (len(expert) != len(model)):
        raise Exception("Count of expert and model ratings are not equal!")
    else:
        y_true = []
        y_pred = []

        for item in expert:
            if item > 2.8:
                y_true.append(1)
            else:
                y_true.append(0)

        for item in model:
            if item > 0.6:
                y_pred.append(1)
            else:
                y_pred.append(0)

        score = accuracy_score(y_true, y_pred)
        return score


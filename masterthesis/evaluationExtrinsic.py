import sys
import json
from gensim import models


def execute():
    pathGoogle = "GoogleNews-vectors-negative300.bin"
    model = models.KeyedVectors.load_word2vec_format(pathGoogle, binary=True)

    #pathEval = "/home/hiwi/Dokumente/masterthesis/data/terminology/eval_extrinsic.json"
    pathEval = "eval_extrinsic.json"
    evalList = json.loads(open(pathEval).read())["eval_list"]

    pathResults = "results.csv"
    results = open(pathResults, "a")
    newLine = "term1;term2;score\n"
    results.write(newLine)

    for i in range(0, len(evalList)-1):
        term1 = evalList[i]
        for j in range(i+1, len(evalList)):
            term2 = evalList[j]

            try:
                score = model.similarity(term1, term2)
                newLine = [term1, term2, str(score)]
                newLine = ";".join(newLine)
                newLine = newLine + "\n"
                results.write(newLine)

            except:
                print("no match for: ", term1, term2)

    results.close()




if __name__ == '__main__':
    execute()
    sys.exit(0)
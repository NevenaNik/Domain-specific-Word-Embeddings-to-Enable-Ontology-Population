import os
import sys

from functions.config import path_trained


"""
def execute():
    models = os.listdir(path_trained)
    aggreg = ["weighted", "majorityWeighted", "majorityUnweighted"]

    for model in models:
        for mode in aggreg:
            os.system(f"python evaluation.py -m {model} -a {mode}")
            os.system(f"python evaluation.py -m {model} -a {mode} --devCheck")

    for model in models:
        for mode in aggreg:
            os.system(f"python evaluation.py -m {model} -a {mode} --spearman")
            os.system(f"python evaluation.py -m {model} -a {mode} --devCheck --spearman")
        
"""

def execute():
    reduction = "lemma"
    rare = "True"
    shuffle = "True"
    ngrams = "training"

    vs = [300, 350, 400, 450, 500]
    win = [10]
    mc = [5]

    pathTrain = "/home/hiwi/Dokumente/masterthesis/masterthesis/trainingLog.csv"
    trainLog = open(pathTrain, "a")
    
    i = 1
    for size in vs:
        for item in win:
            for count in mc:
                """
                # Combination 1:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count}')
                newline1 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "CBOW", "softmax"]
                newline1 = ";".join(newline1)
                newline1 = newline1 + "\n"
                trainLog.write(newline1)

                model = f"02w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg0_hs1.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1
                """
                # Combination 2:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --skipgram')
                newline2 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "skipgram", "softmax"]
                newline2 = ";".join(newline2)
                newline2 = newline2 + "\n"
                trainLog.write(newline2)
                
                model = f"02w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg1_hs1.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1
                """
                # Combination 3:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --negSampl')
                newline3 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "CBOW", "negSampl"]
                newline3 = ";".join(newline3)
                newline3 = newline3 + "\n"
                trainLog.write(newline3)
                
                model = f"02w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg0_hs0.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1

                # Combination 4:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --skipgram --negSampl')
                newline4 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "skipgram", "negSampl"]
                newline4 = ";".join(newline4)
                newline4 = newline4 + "\n"
                trainLog.write(newline4)
                
                model = f"02w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg1_hs0.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1
                """
    trainLog.close()



if __name__ == '__main__':
    execute()
    sys.exit(0)
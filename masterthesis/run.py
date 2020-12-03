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

    vs = [500]
    win = [10]
    mc = [5]
    percentage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    pathTrain = "/home/hiwi/Dokumente/masterthesis/masterthesis/trainingLog.csv"
    trainLog = open(pathTrain, "a")
    
    i = 1
    for size in vs:
        for item in win:
            for count in mc:
                for perc in percentage:
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
                    os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --skipgram -p {perc}')
                    newline2 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "skipgram", "softmax", str(perc)]
                    newline2 = ";".join(newline2)
                    newline2 = newline2 + "\n"
                    trainLog.write(newline2)
                
                    #model = f"01size{perc}_w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg1_hs1.bin"
                    #os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
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

def evaluate():
    vs = [500]
    win = [10]
    mc = [5]
    hs = [1, 0]
    sg = [0, 1]
    percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sentLen = [1, 5, 10, 50, 100, 250, 500, 600, 700, 1000, 1500, 2000, "Complete"]
    shuffled = [True, False]

    """
    i = 1
    for size in vs:
        for item in win:
            for count in mc:
                for perc in percentage:
                    model = f"02size{perc}_w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg1_hs1.bin"
                    os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                    i += 1
    
    i = 1
    for size in vs:
        for item in win:
            for count in mc:
                for method in hs:
                    for algo in sg:
                        model = f"w2v_ngramsTraining_vs{size}_win{item}_mc{count}_sg{algo}_hs{method}.bin"
                        os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                        i += 1             
    """
    i = 1
    for length in sentLen:
        for option in shuffled:
            model = f"03sentence{length}_w2v_shuffled{option}_ngramsTraining_vs500_win10_mc5_sg1_hs1.bin"
            os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
            i += 1






if __name__ == '__main__':
    #execute()
    evaluate()
    sys.exit(0)
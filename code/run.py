import os
import sys

from functions.config import path_models




def execute():

    # Use to train models
    reduction = "lemma"
    rare = "True"
    shuffle = "True"
    ngrams = "training"

    vs = [500]
    win = [10]
    mc = [5]
    skipgram = [True]
    negSampl = [False]

    percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    trainLog = open("modeltraining.csv", "a")
    
    i = 1
    for size in vs:
        for item in win:
            for count in mc:
                   
                # Combination 1:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count}')
                newline1 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "CBOW", "softmax"]
                newline1 = ";".join(newline1)
                newline1 = newline1 + "\n"
                trainLog.write(newline1)

                model = f"w2v_{reduction}_rare{rare}_ngrams{ngrams}_shuffled{shuffle}_vs{size}_win{item}_mc{count}_sg0_hs1_size{self.percentage}.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1
                    
                # Combination 2:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --skipgram -p {perc}')
                newline2 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "skipgram", "softmax", str(perc)]
                newline2 = ";".join(newline2)
                newline2 = newline2 + "\n"
                trainLog.write(newline2)
                
                model = f"w2v_{reduction}_rare{rare}_ngrams{ngrams}_shuffled{shuffle}_vs{size}_win{item}_mc{count}_sg1_hs1_size{self.percentage}.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1
                
                # Combination 3:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --negSampl')
                newline3 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "CBOW", "negSampl"]
                newline3 = ";".join(newline3)
                newline3 = newline3 + "\n"
                trainLog.write(newline3)
                
                model = f"w2v_{reduction}_rare{rare}_ngrams{ngrams}_shuffled{shuffle}_vs{size}_win{item}_mc{count}_sg0_hs0_size{self.percentage}.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1

                # Combination 4:
                os.system(f'python training.py --rare --shuffle -vs {size} -win {item} -mc {count} --skipgram --negSampl')
                newline4 = [str(i), reduction, rare, shuffle, ngrams, str(size), str(item), str(count), "skipgram", "negSampl"]
                newline4 = ";".join(newline4)
                newline4 = newline4 + "\n"
                trainLog.write(newline4)

                model = f"w2v_{reduction}_rare{rare}_ngrams{ngrams}_shuffled{shuffle}_vs{size}_win{item}_mc{count}_sg1_hs0_size{self.percentage}.bin"
                os.system(f'python evaluation.py --id {i} -m {model} -a majorityWeighted --devCheck')
                i += 1

    
    trainLog.close()


if __name__ == '__main__':
    
    execute()
    
    sys.exit(0)
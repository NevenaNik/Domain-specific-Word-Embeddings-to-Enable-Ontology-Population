def vote_aggreg(method, dictionary):
    
    votes = {}

    focus_term = list(dictionary.keys())[0]
    dictionary = dictionary[focus_term]

    if method == "weighted":
        for term in dictionary.keys():
            vote = 0
            count = 0
            for item in dictionary[term]:
                vote = vote + item[0] * (5-item[1])
                count = count + (5- item[1])
            vote = vote / count
            votes[term] = vote
    elif method == "majority_w":
        for term in dictionary.keys():
            counts = {"1":0, "2":0, "3":0, "4":0}
            for item in dictionary[term]:
                if item[0] == 1:
                    counts["1"] = counts["1"] + (5-item[1])
                elif item[0] == 2:
                    counts["2"] = counts["2"] + (5-item[1])
                elif item[0] == 3:
                    counts["3"] = counts["3"] + (5-item[1])
                else:
                    counts["4"] = counts["4"] + (5-item[1])
            itemMaxValue = max(counts.items(), key=lambda x: x[1])
            listOfKeys = list()
            for key, value in counts.items()    :
                if value == itemMaxValue[1]:
                    listOfKeys.append(int(key))
            vote = sum(listOfKeys) / len(listOfKeys)
            votes[term] = vote            
    else:
        for term in dictionary.keys():
            counts = {"1":0, "2":0, "3":0, "4":0}
            for item in dictionary[term]:
                if item[0] == 1:
                    counts["1"] += 1
                elif item[0] == 2:
                    counts["2"] += 1
                elif item[0] == 3:
                    counts["3"] += 1
                else:
                    counts["4"] += 1
            itemMaxValue = max(counts.items(), key=lambda x: x[1])
            listOfKeys = list()
            for key, value in counts.items()    :
                if value == itemMaxValue[1]:
                    listOfKeys.append(int(key))
            vote = sum(listOfKeys) / len(listOfKeys)
            votes[term] = vote  

    return votes
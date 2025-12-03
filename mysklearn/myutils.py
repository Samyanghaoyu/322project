# myutils.py

def majority_vote(labels):
    from collections import Counter
    return Counter(labels).most_common(1)[0][0]

from __future__ import division

'''
 -correct_pred: The correct prediction
 -predictions: List of the models predictions, sorted from most likely ([0]) to least ([-1])
 
 If the correct_pred is not in predictions, the rank is set to 0
'''
def calculate_MRR(correct_pred, predictions):
    ranks = []
    for i in range(len(correct_pred)):
        ranks.append(get_reciprocal_rank(correct_pred[i], predictions[i]))
    return sum(ranks)/len(ranks)


def get_reciprocal_rank(correct, predictions):
    rank = get_first_occurance(correct, predictions)

    if rank == -1:
        return 0
    else:
        return 1/(rank+1)

def get_first_occurance(item, array):
    for i in xrange(array.shape[0]):
        if array[i] == item:
            return i
    return -1

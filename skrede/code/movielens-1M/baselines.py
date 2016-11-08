from __future__ import division

# Use a map to store which sessions each item appears in
# Use this to calculate knn
def print_baseline_knn(k, train_set, test_set):
    # find which sessions each item appears in
    item_session_map = {}
    for i in range(len(train_set)):
        items = train_set[i]
        for item in items:
            if item not in item_session_map:
                item_session_map[item] = []
            item_session_map[item].append(i)

    # test the prediction
    correct_pred_count = 0
    total_pred_count = 0
    for test_session in test_set:
        for i in range(len(test_session)-1):
            x = test_session[i]
            y = test_session[i+1]
            predictions = get_knn_predictions(k, x, item_session_map, train_set)
            if y in predictions:
                correct_pred_count += 1
            total_pred_count += 1

    accuracy = correct_pred_count/total_pred_count
    print "accuracy of k-nn baseline: " + str(accuracy) + " | k="+str(k)

# TODO: faster to store a matrix with all possible values
def get_knn_predictions(k, x, item_session_map, session_list):
    # 3952 = total number of evaluated movies
    item_cooccurances = [0]*3952
    if x not in item_session_map:
        return [-1]
    x_occurances = item_session_map[x]
    for x_session in x_occurances:
        for movie in session_list[x_session]:
            item_cooccurances[movie] += 1

    # Don't recommend the same movie as the input
    item_cooccurances[x] = -1
    
    return sorted(range(len(item_cooccurances)), key=lambda i:item_cooccurances[i])[-k:]


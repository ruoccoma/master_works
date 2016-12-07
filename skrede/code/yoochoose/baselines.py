from __future__ import division

def print_all_baslines_precomputed():
    print(" --------------------- Baselines ------------------------------")
    print("| accuracy of random guessing should be about: 0.00296120817293 |")
    print("| accuracy of top-k baseline: 0.0283920615634                   |")
    print("| accuracy of k-nn baseline: 0.0592200428166 | k=10             |")
    print(" ---------------------------------------------------------------")

def print_all_baselines(k, n_classes, test_set, top_k, knn_training_data):
    print("###### BASELINES ##########")
    print_baseline_random(k, n_classes)
    print_baseline_top_k(top_k, test_set)
    print_baseline_knn(k, knn_training_data, test_set)
    print("###########################")


def print_baseline_random(k, n_classes):
    accuracy = k/n_classes
    print("| accuracy of random guessing should be about: "+str(accuracy))

def print_baseline_top_k(top_k_movies, test_data):
    correct_pred = 0
    count = 0
    for session in test_data:
        for movie in session[1:]:
            if movie in top_k_movies:
                correct_pred += 1
            count += 1
    
    accuracy = correct_pred/count
    print("| accuracy of top-k baseline: " + str(accuracy))

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
    print("accuracy of k-nn baseline: " + str(accuracy) + " | k="+str(k))

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


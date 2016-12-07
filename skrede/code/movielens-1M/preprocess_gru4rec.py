# -*- coding: utf-8 -*-

import sys

base_path = '/home/ole/recSys-pro/movielens-1M/'
data_set = base_path + 'ratings.dat'
preprocess_log = base_path + 'preprocess.log'

buffer_size = 20000000
n_movies = 3952

# most are below 900
max_length = 10
n_top_movies = 10


def preprocess_data(in_file, train_file, test_file, train_knn_file):
    global preprocess_log
    movie_statistics = [0]*3952
    gru4rec = "_gru4rec"

    train_file = base_path + train_file + gru4rec
    test_file = base_path + test_file + gru4rec
    train_knn_file = base_path + train_knn_file + gru4rec
    preprocess_log = preprocess_log + gru4rec

    data = {}

    # Read all data first
    with open(in_file, buffering=buffer_size) as in_f:
        for line in in_f:
            line = line.split('::')
            line = map(int, line)
            
            # -1 to convert to 0-indexing
            user_id   = line[0]-1
            movie_id  = line[1]-1
            rating    = line[2]-1
            timestamp = line[3]

            if user_id not in data:
                data[user_id] = []

            data[user_id].append([movie_id, timestamp])

            movie_statistics[movie_id] += 1

    # Sort the user-clicks by time. Only sorts per user.
    for key in data.keys():
        data[key] = sorted(data[key], key=lambda x: x[1])

    # Only keep movie ratings with good enough support. 
    # (MovieID should appear at least 5 times)
    # I also only keep the movie_id info from here on
    filtered_data = []
    for key in data.keys():
        session = data[key]
        new_session = []

        for rating_info in session:
            movie_id = rating_info[0]
            movie_id_support = movie_statistics[movie_id]

            # Only keep movies with enough support
            if 5 < movie_id_support:
                new_session.append(rating_info)

        # Only keep sessions that are long enough        
        if max_length <= len(new_session):
            filtered_data.append(new_session)

    # Free up RAM by removing old data
    data = filtered_data
    filtered_data = None

    # Check how many movie_ids we are left with
    remaining_movie_ids = {}
    for session in data:
        for rating_info in session:
            movie_id = rating_info[0]
            if movie_id not in remaining_movie_ids:
                remaining_movie_ids[movie_id] = len(remaining_movie_ids.keys())
    
    # Replace the original movie_ids with lower values so we don't use higher values than
    #  necessary
    for session_index in range(len(data)):
        session = data[session_index]
        for rating_index in range(len(session)):
            session[rating_index][0] = remaining_movie_ids[session[rating_index][0]]
    
    # Update total number of movies we are left with
    n_movies = len(remaining_movie_ids.keys())

    # Split data between training and testing
    training_split = int(0.8*len(data))
    training_data = data[:training_split]
    test_data = data[training_split:]
    
    # Update movie statistics
    movie_statistics = [0]*n_movies
    for session in training_data:
        for rating in session:
            movie_statistics[rating[0]] += 1

    # Find top k movies
    top_k_movies = sorted(range(len(movie_statistics)), key=lambda i: movie_statistics[i])[-n_top_movies:]

    # Store the current training set as is for the k-nn baseline to use
    with open(train_knn_file, 'w') as out_f:
        for session in training_data:
            line = ""
            for rating in session:
                movie_id = rating[0]
                line += str(movie_id) + " "
            out_f.write(line.rstrip()+'\n')
    

    # Create multiple sessions out of each session.
    # E.g. [1, 3, 7, 2, 4] -> {[1, 3, 7], [3, 7, 2], [7, 2, 4]} if max_length == 3
    processed_sessions = []
    for session in training_data:
        length = len(session) + 1
        i = 0
        while i+max_length <= length:
            new_session = session[i:i+max_length]
            processed_sessions.append(new_session)
            i += 2

    training_data = processed_sessions

    # Cut the test sessions to shorter sessions also (here we only split them to be shorter than max_length)
    processed_sessions = []
    for session in test_data:
        session
        while max_length < len(session):
            tmp = session[:max_length]
            processed_sessions.append(tmp)
            session = session[max_length:]
        if 1 < len(session):
            processed_sessions.append(session)

    test_data = processed_sessions
    
    with open(train_file, 'w') as out_f:
        out_f.write("SessionId\tItemId\tTime\n")
        for session_id in range(len(training_data)):
            line = ""
            session = training_data[session_id]
            for rating in session:
                movie = rating[0]
                timestamp = rating[1]
                out_f.write(str(session_id)+"\t"+str(movie)+"\t"+str(timestamp)+"\n")

    with open(test_file, 'w') as out_f:
        out_f.write("SessionId\tItemId\tTime\n")
        for session_id in range(len(test_data)):
            line = ""
            session = test_data[session_id]
            for rating in session:
                movie = rating[0]
                timestamp = rating[1]
                out_f.write(str(session_id)+"\t"+str(movie)+"\t"+str(timestamp)+"\n")
            
    # log some info abut the processing (useful to check correctness)
    with open(preprocess_log, 'w') as log:    
        log.write("max_length (session):"+str(max_length)+"\n")
        log.write("num movies:"+str(n_movies)+"\n")
        log.write("training_split:"+str(training_split)+"\n")
        log.write("training_sessions:"+str(len(training_data))+"\n")
        log.write("test_sessions:"+str(len(test_data))+"\n")
        log.write("top_k_movies:"+str(top_k_movies)+"\n")
    
preprocess_data(data_set, "1M-train.dat", "1M-test.dat", "1M-train-knn.dat")


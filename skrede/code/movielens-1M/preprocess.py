# -*- coding: utf-8 -*-

import sys

base_path = '/home/ole/recSys-pro/movielens-1M/'
data_set = base_path + 'ratings.dat'
preprocess_log = base_path + 'preprocess.log'

buffer_size = 20000000
n_movies = 3952

# most are below 900
max_length = 20
n_top_movies = 20

def preprocess_data(in_file, train_file, test_file):

    movie_statistics = [0]*3952
    
    train_file = base_path + train_file
    test_file = base_path + test_file
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

            data[user_id].append([movie_id, rating, timestamp])

            movie_statistics[movie_id] += 1

	# Sort the user-clicks by time. Only sorts per user.
	for key in data.keys():
		data[key] = sorted(data[key], key=lambda x: x[2])


	# Clip the long sessions
        new_sessions = []
	for key in data.keys():
		user_ratings = data[key]
		if max_length<len(user_ratings):
			data[key] = user_ratings[:max_length]
                        rest = user_ratings[max_length:]
                        while max_length < len(rest):
                            tmp = rest[:max_length]
                            rest = rest[max_length:]
                            new_sessions.append(tmp)
                        if 1 < len(rest):
                            new_sessions.append(rest)

        for new_sess in new_sessions:
            data[len(data.keys())] = new_sess


        training_split = int(0.8*len(data.keys()))
	
	with open(train_file, 'w') as out_f:
		for key in range(0, training_split):
			line = ""
			for rating in data[key]:
				line += str(rating[0]) + " "
			out_f.write(line.rstrip()+'\n')

	with open(test_file, 'w') as out_f:
		for key in range(training_split, len(data.keys())):
			line = ""
			for rating in data[key]:
				line += str(rating[0]) + " "
			out_f.write(line.rstrip()+'\n')
	
        # Find top k movies
        top_k_movies = sorted(range(len(movie_statistics)), key=lambda i: movie_statistics[i])[-n_top_movies:]
			
	# log some info abut the processing (useful to check correctness)
	with open(preprocess_log, 'a') as log:
	    log.write("\n ----------------- \n")
	    #log.write("longest session: "+str(longest_session)+"\n")
	    #log.write("number of sessions: "+str(num_sessions)+"\n")
	    log.write("training_sessions: "+str(training_split-1)+"\n")
	    log.write("test_sessions: "+str(len(data.keys())-training_split+1)+"\n")
            log.write("training_split: "+str(training_split)+"\n")
            log.write("max_length (session): "+str(max_length)+"\n")
            log.write("top_k_movies: "+str(top_k_movies)+"\n")
	


preprocess_data(data_set, "1M-train.dat", "1M-test.dat")


# -*- coding: utf-8 -*-

import sys

base_path = '/home/ole/recSys-pro/movielens-1M/'
data_set = base_path + 'ratings.dat'
preprocessed_data = base_path + '1m-training_processed.dat'
preprocess_log = base_path + 'preprocess.log'

buffer_size = 20000000

n_users = 6040
n_movies = 3952

# Most sessions contain less than 500 ratings, so we cut off there.
max_length = 500


def preprocess_data(in_file, out_file):
	data = {}

	# Read all data first
	with open(in_file, buffering=buffer_size) as in_f:
		for line in in_f:
			line = line.split('::')
			line = map(int, line)

			user_id   = line[0]
			movie_id  = line[1]
			rating    = line[2]
			timestamp = line[3]

			if 3 < rating:
				if user_id not in data:
					data[user_id] = []

				data[user_id].append([movie_id, rating, timestamp])

	# Sort the user-clicks by time. Only sorts per user.
	for key in data.keys():
		data[key] = sorted(data[key], key=lambda x: x[2])
 
	# Clip the long sessions
	for key in data.keys():
		user_ratings = data[key]
		if max_length<len(user_ratings):
			data[key] = user_ratings[:max_length]

	longest_session = 0
	num_sessions = 0
	with open(out_file, 'w') as out_f:
		for key in data.keys():
			num_sessions += 1
			if longest_session < len(data[key]):
				longest_session = len(data[key])
			for rating in data[key]:
				out_f.write(str(key)+':'+str(rating[0])+':'+str(rating[1])+':'+str(rating[2])+'\n')
			
	# log some info abut the processing (useful to check correctness)
	with open(preprocess_log, 'a') as log:
	    log.write("\n ----------------- \n")
	    log.write("longest session: "+str(longest_session)+"\n")
	    log.write("number of session: "+str(num_sessions)+"\n")


preprocess_data(data_set, preprocessed_data)


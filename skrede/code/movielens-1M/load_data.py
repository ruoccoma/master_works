import time
import numpy as np

def load_data_set(data_file):
	print " loading data set:" + data_file

	# Measure how long time it takes to load the entire data file
	start = time.time()

	# Store the sessions in a list
	data = []

	# Read the data line by line
	with open(data_file, 'r') as f:
	    for line in f:

	    	# Convert the line to a list of item/movie/click-IDs
        	line = line.split(' ')
        	line = map(int, line)
        	data.append(line)

	runtime = time.time() - start

	# When creating batches, we need to know how many samples we have to work with
	n_sessions = len(data)

	print "   data loaded in " + str(runtime) + "s"

	return data, n_sessions

'''
Create n vectors of size n, like so:
[
 [1, 0, 0, 0, ....]
 [0, 1, 0, 0, ....]
 [0, 0, 1, 0, ....]
 [0, 0, 0, 1, ....]
 ...
 [0, 0, ...., 0, 1]
]

This saves time by using space. Can just point to these when needed.
'''
def create_1_HOT_vectors(n):
	print " creating library of 1-HOT vectors..."

	# Measure the runtime to create the vectors
	start = time.time()

	# Create a list of vectors
	one_hots = []

	# n vectors of size n
	for i in range(n):
	    tmp = [0.]*n
	    tmp[i] = 1.0
	    one_hots.append(np.array(tmp))

	# Sessions shorter than the max length needs to be padded with a dummy vector
	dummy_vector = [0.]*n

	runtime = time.time() - start
	print "   1-HOT vectors created in " + str(runtime) + "s"

	return one_hots, dummy_vector
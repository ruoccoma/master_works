import numpy as np
import pickle
import sys

'''
-Session ID – the id of the session. In one session there are one or many clicks. Could be represented as an integer number.
-Timestamp – the time when the click occurred. Format of YYYY-MM-DDThh:mm:ss.SSSZ
-Item ID – the unique identifier of the item that has been clicked. Could be represented as an integer number.
-Category – the context of the click. The value "S" indicates a special offer, "0" indicates  a missing value, a number between 1 to 12 indicates a real category identifier,
 any other number indicates a brand. E.g. if an item has been clicked in the context of a promotion or special offer then the value will be "S", if the context was a brand i.e BOSCH,
 then the value will be an 8-10 digits number. If the item has been clicked under regular category, i.e. sport, then the value will be a number between 1 to 12. 
'''

raw_file = 'yoochoose-clicks.dat'
out_file = 'processed_data.csv'
n_lines_to_read = 10000

# Read up to n_lines_to_read of lines from the dataset and put sessions in a dictionary
def gather_sessions():
	sessions = {}	
	with open(raw_file, buffering=200000) as f:
		try:
			lines_read = 0
			for line in f:
				# use a limit for number of lines to read
				if lines_read == n_lines_to_read:
					break

				session_id = line.split(',')[0]
				item_id = line.split(',')[2]

				if session_id in sessions:
					sessions[session_id].append(item_id)
				else:
					sessions[session_id] = [item_id]

				lines_read += 1
		except Exception as e:
			print(str(e))
	return sessions

sessions = gather_sessions()
print("-------------------------------------------")
print("Number of sessions before filtering:", len(sessions.keys()))

# Remove sessions with only one click
def filter_sessions():
	filtered_sessions = {}
	for k in sessions.keys():
		if len(sessions[k]) > 1:
			filtered_sessions[k] = sessions[k]

	return filtered_sessions


sessions = filter_sessions()
print("Number of sessions:",len(sessions.keys()))

# Creates a dictionary with all items and an index in a dictionary. Indexes start with 0 and upwards.
# This lets us map the items to vectors
def create_item_map():
	item_map = {}
	for k in sessions.keys():
		v = sessions[k]
		for item in v:
			if item not in item_map:
				item_map[item] = len(item_map.keys())

	return item_map

item_map = create_item_map()
print("Number of items:", len(item_map.keys()))

def convert_items_to_vec():
	n_items = len(item_map.keys())
	item_vec = {}

	for k in item_map.keys():
		vec = np.zeros(n_items)
		index = int(item_map[k])
		vec[index] = 1
		vec = list(vec)
		item_vec[k] = vec

	return item_vec

item_vec = convert_items_to_vec()

def convert_sessions_to_vec():
	count = 0
	session_vec = []
	for k in sessions.keys():
		session = sessions[k]
		vec_session = []
		for i in session:
			vec_session.append(item_vec[i])
			count += 1
		session_vec.append(vec_session)

	print("Total number of items in sessions:", count)
	print("Size of an item vector in bytes:", sys.getsizeof(session_vec[0][0]))
	bytes_per_vector = sys.getsizeof(session_vec[0][0])
	total_bytes_used_on_item_vectors = bytes_per_vector * count
	KB = total_bytes_used_on_item_vectors/1024
	MB = KB/1024
	print("MB used on item vectors:", MB)
	return session_vec

session_vec = convert_sessions_to_vec()
print("Number of sessions in vec format:",len(session_vec))

with open('sessions.pickle', 'wb') as f:
	pickle.dump(session_vec, f)

with open('read-lines.txt', 'w') as f:
	f.write(str(n_lines_to_read)+'\n')

print("-------------------------------------------")
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

training_clicks_file = 'yoochoose-clicks.dat'
processed_training_data_file = 'processed_training.txt'
items_file = 'processed_data_items.txt'

'''
Read the dataset file. Count the number of items and the number of sessions. Write each session
to one line in the output file. Each line consists of the items in that session, in the original
order.
'''
def process_sessions(in_file, out_file, store_num_items=True):
	n_sessions = 0
	item_clicks = {}
	session_id = -1

	with open(in_file, buffering=200000) as f:
		with open(out_file, 'w') as of:
			try:
				# read first session id
				line = f.readline()
				session_id = line.split(',')[0]
				f.seek(0)

				session = ""
				session_length = 0
				for line in f:
					new_session_id = line.split(',')[0]
					item_id = line.split(',')[2]

					# Collection of all item_ids that are used
					if item_id not in item_clicks:
						item_clicks[item_id] = len(item_clicks.keys())

					# update the items in the session or write out old session and start on new
					if new_session_id==session_id:
						session += " " + item_id
						session_length += 1
					else:
						if session_length > 1:
							of.write(session +'\n')
						session = " " + item_id
						session_id = new_session_id
						session_length = 1


			except Exception as e:
				print(str(e))

	if store_num_items:
		with open(items_file, 'w') as f:
			

process_sessions(training_clicks_file, processed_training_data_file)
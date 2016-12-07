# -*- coding: utf-8 -*-

import sys
import pickle

base_path = '/home/ole/recSys-pro/yoochoose/'
preprocess_log = base_path + 'preprocess.log'

hidasi_full_train = base_path + 'rsc15_train_full.txt'
ole_full_train = base_path + 'rsc15_train_full_ole.txt'

hidasi_test = base_path + 'rsc15_test.txt'
ole_test = base_path + 'rsc15_test_ole.txt'

meta_pickle = base_path + 'meta.pickle'

buffer_size = 20000000
max_length = 10
k = 20

def read_data(in_file):
    data = {}
    # Read all the sessions into the data map
    with open(in_file, buffering=buffer_size) as in_f:
        for line in in_f:
            if line[0] == 'S':
                continue
       
            line = line.split('\t')
            session_id = int(line[0])
            item_id = int(line[1])
            # line[0]=sessionID   line[1]=itemID   line[2]=timestamp
            if session_id not in data:
                data[session_id] = []
            data[session_id].append(item_id)

    return data

def write_data(out_file, data):
    with open(out_file, 'w') as out_f:
        for key in data:
            session = data[key]
            session = session[:max_length]
            line = ""
            for item_click in session:
                line += str(item_click) + " "
            out_f.write(line.rstrip()+"\n")


# We use the preprocessing by Hidasi to get the correct sessions (in format: sessionID | itemID | timestamp)
#  then we do another pass to get the data in a format that is easier to work with
def preprocess_data():
    train_data = read_data(hidasi_full_train)
    test_data = read_data(hidasi_test)

    session_lengths = [0]*201

    # Go through all the itemIDs and map them down to smallest possible values
    all_items = {}
    for key in train_data:
        session = train_data[key]
        session_lengths[len(session)] += 1
        for item_click in range(len(session)):
            item = session[item_click]
            if item not in all_items:
                all_items[item] = len(all_items.keys())
            session[item_click] = all_items[item]
    for key in test_data:
        session = test_data[key]
        session_lengths[len(session)] += 1
        for item_click in range(len(session)):
            item = session[item_click]
            if item not in all_items:
                all_items[item] = len(all_items.keys())
            session[item_click] = all_items[item]

    n_items = len(all_items.keys())

    write_data(ole_full_train, train_data)
    write_data(ole_test, test_data)

    with open(preprocess_log, 'w') as log:
        log.write("max_length (session):"+str(max_length)+"\n")
        log.write("num items:"+str(n_items)+"\n")

    print("Session lengths:")
    print(session_lengths)
    print("Num sessions:")
    print(sum(session_lengths))
    print("Num sessions@10:")
    print(sum([session_lengths[i] for i in range(11)]))
    print("Num sessions@20:")
    print(sum([session_lengths[i] for i in range(21)]))
    print("Num items:")
    print(n_items)
    
    # Store some metadata
    meta = {'n_classes':n_items,
    	    'max_sequence_length':max_length,
            'k': k
            }
    pickle.dump( meta, open( meta_pickle, "wb" ) )



preprocess_data()


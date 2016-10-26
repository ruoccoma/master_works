# -*- coding: utf-8 -*-

import sys

base_path = '/home/ole/recSys-pro/yoochoose/'
preprocess_log = base_path + 'preprocess.log'

hidasi_full_train = base_path + 'rsc15_train_full.txt'
ole_full_train = base_path + 'rsc15_train_full_ole.txt'

hidasi_test = base_path + 'rsc15_test.txt'
ole_test = base_path + 'rsc15_test_ole.txt'

buffer_size = 20000000

session_max_length = 19

def crop_session(session):
    session = session.split(' ')
    out = ""
    for i in range(session_max_length):
        out += session[i] + " "
    return out, session_max_length

# We use the preprocessing by Hidasi to get the correct sessions (in format: sessionID | itemID | timestamp)
#  then we do another pass to get the data in a format that is easier to work with
def preprocess_data(in_file, out_file):
    # all items
    items = {}
    session_id = -1
    n_sessions = 0
    longest_session = 0
    with open(in_file, buffering=buffer_size) as in_f:
        with open(out_file, 'w') as out_f:
            try:
                # read first session id, read twice to skip the first line
                line = in_f.readline()

                line = in_f.readline()
                line = line.split('\t')

                session_id = int(line[0])

                in_f.seek(0)
                line = in_f.readline() # skip first line again

                session = ""
                session_length = 0

                for line in in_f:
                    line = line.split('\t')
                    tmp_session_id = int(line[0])
                    item_id = int(line[1])

                    # Map item ids down to smaller numbers
                    if item_id not in items:
                        items[item_id] = len(items.keys())
                    # use the downscaled item_id
                    item_id = items[item_id]

                    # Are we still on the same session?
                    if tmp_session_id == session_id:
                        session += str(item_id) + " "
                        session_length += 1
                    else:
                        if 19 < session_length:
                            session, session_length = crop_session(session)
                        # write out this session
                        out_f.write(session.rstrip() + '\n')
                        if longest_session < session_length:
                            longest_session = session_length
                        n_sessions += 1
                        # start next session
                        session = str(item_id) + " "
                        session_length = 1
                        session_id = tmp_session_id

            except Exception as e:
                print str(e)

    # log some info abut the processing (useful to check correctness)
    with open(preprocess_log, 'a') as log:
        log.write("\n ----------------- \n")
        log.write("  Processed "+in_file+" and stored result in "+out_file+"\n")
        log.write("  Number of sessions: "+str(n_sessions)+"\n")
        log.write("  Number of unique items: "+str(len(items.keys()))+"\n")
        log.write("  Longest session: "+str(longest_session)+"\n")


preprocess_data(hidasi_full_train, ole_full_train)


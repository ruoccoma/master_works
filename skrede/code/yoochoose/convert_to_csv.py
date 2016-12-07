import os
import pickle

location = '/home/ole/recSys-pro/yoochoose/'
test_file  = 'rsc15_test_ole.txt'
train_file = 'rsc15_train_full_ole.txt'
n_train_files = 4
test_csv   = location + 'rsc15_test.csv'
meta_file  = location + 'meta.pickle'

meta = pickle.load( open(meta_file, 'rb'))
sequence_length = meta['max_sequence_length']
pickle.dump( meta, open(meta_file, 'wb'))

def process_line(line):
    sequence = line.replace("\n", "")
    sequence = sequence.split(" ")
    seq_len = len(sequence)

    # Pad the sequence
    while len(sequence)<sequence_length:
        sequence.append('0')
    
    # Add the sequence length as the first value
    sequence.insert(0, seq_len)

    # Replace spaces with commas (for .csv)
    sequence = ",".join(str(i) for i in sequence)
    sequence = sequence + "\n"

    return sequence

def convert_test_to_csv(location, in_file):
    out_file = in_file.split('.')
    out_file = out_file[0] + '.csv'
    out_file = location + out_file
    in_file = location + in_file

    print("Converting", in_file, "\n        to", out_file)
    
    with open(out_file, 'w') as out_f:
        with open(in_file, 'r') as in_f:
            for line in in_f:
                line = process_line(line)
                out_f.write(line)

def convert_train_to_csv(location, train_file, n_train_files):
    # Create directory for the splitted training files
    csv_dir = location + 'train_split/'
    if not os.path.isdir(csv_dir):
        os.mkdir(csv_dir)
    
    # Create paths for the training files (.csv)
    train_files = [csv_dir+'train'+str(i)+'.csv' for i in range(n_train_files)]

    train_file = location + train_file
    file_index = 0
    files = [open(f, 'w') for f in train_files]
    with open(train_file, 'r') as in_f:
        for line in in_f:
            line = process_line(line)
            files[file_index].write(line)
            file_index += 1
            file_index = file_index % len(train_files)
    for f in files:
        f.close()

    meta = pickle.load( open(meta_file, 'rb'))
    meta['train_files'] = train_files
    pickle.dump( meta, open(meta_file, 'wb'))

def convert_all():
    convert_test_to_csv(location, test_file)
    convert_train_to_csv(location, train_file, n_train_files)


convert_all()


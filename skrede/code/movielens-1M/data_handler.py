def load_data_set(data_file):
    print(" loading data set:" + data_file)

    # Store the sessions in a list
    data = []

    # Read the data line by line
    with open(data_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.split(',')
            line = [int(i) for i in line]
            
            seq_len = line[0]
            features = line[1:-1]
            targets  = line[2:]
            mask     = [0]*len(features)
            for i in range(seq_len):
                mask[i] = 1

            data.append([seq_len, features, targets, mask])

    return data
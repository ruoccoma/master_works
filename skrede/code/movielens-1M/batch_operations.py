from load_data import *

class DatasetManager:

    def __init__(self, training_data_file, test_data_file, n_classes, max_length, batch_size):
        self.training_data_file = training_data_file
        self.test_data_file = test_data_file
        self.n_classes = n_classes
        self.max_length = max_length
        self.batch_size = batch_size

        # Read data
        self.training_data, self.n_training_sessions = load_data_set(training_data_file)
        self.test_data, self.n_test_sessions = load_data_set(test_data_file)

        # Create 1-HOT vectors in advance. Space/time tradeoff
        self.one_hots, self.dummy_vector = create_1_HOT_vectors(n_classes)

        self.training_batch_current_index = 0
        self.test_batch_current_index = 0

    def get_number_of_training_batches(self):
        # -(-) gives division rounded up
        return -(-self.n_training_sessions/self.batch_size)

    def get_number_of_test_batches(self):
        # -(-) gives division rounded up
        return -(-self.n_test_sessions/self.batch_size)

    def get_next_training_batch(self):
        training_batch_x = []
        training_batch_y = []
        training_batch_lengths = []
        batch_index_goal = self.training_batch_current_index + self.batch_size

        while self.training_batch_current_index < self.n_training_sessions and self.training_batch_current_index < batch_index_goal:
            tmp_x, tmp_y, tmp_s = self.get_training_session(self.training_batch_current_index)
            training_batch_x.append(tmp_x)
            training_batch_y.append(tmp_y)
            training_batch_lengths.append(tmp_s)
            self.training_batch_current_index += 1
    
        if self.n_training_sessions <= self.training_batch_current_index:
            self.training_batch_current_index = 0

        return training_batch_x, training_batch_y, training_batch_lengths

    def get_next_test_batch(self):
        test_batch_x = []
        test_batch_y = []
        test_batch_lengths = []
        batch_index_goal = self.test_batch_current_index + self.batch_size
    
        while self.test_batch_current_index < self.n_test_sessions and self.test_batch_current_index < batch_index_goal:
            tmp_x, tmp_y, tmp_s = self.get_test_session(self.test_batch_current_index)
            test_batch_x.append(tmp_x)
            test_batch_y.append(tmp_y)
            test_batch_lengths.append(tmp_s)
            self.test_batch_current_index += 1
    
        if self.n_test_sessions <= self.test_batch_current_index:
            self.test_batch_current_index = 0

        return test_batch_x, test_batch_y, test_batch_lengths

    '''
    Takes session number i in the training data and extracts input and target values.
    These are returned as 1-HOT encoded vectors (each itemID).
    '''
    def get_training_session(self, i):
        # the session as natural numbers (IDs)
        raw_data = self.training_data[i]

        # -1 since last one is only target and first one is not target (only input)
        real_session_length = len(raw_data) - 1 

        # lists for input and target vectors
        session_x = [None]*(self.max_length-1)
        session_y = [None]*(self.max_length-1)

        # 0-real_session_length are inputs, 1-real_session_length+1 are targets
        for i in range(real_session_length):
            session_x[i] = self.one_hots[raw_data[i]]
            session_y[i] = self.one_hots[raw_data[i+1]]

        # number of dummy vectors we need to add
        padding_length = self.max_length - len(raw_data)
        for i in range(padding_length):
            # pad both input and targets with dummies
            session_x[real_session_length+i] = self.dummy_vector
            session_y[real_session_length+i] = self.dummy_vector

        return session_x, session_y, real_session_length


    '''
    In the test sessions the session_y contains the actual ids, NOT 1-HOT representations,
    because we are not training, we are comparing the output.
    '''
    def get_test_session(self, i):
        # The session as natural numbers (IDs)
        raw_data = self.test_data[i]

        # -1 since last one is only target and first one is not target (only input)
        real_session_length = len(raw_data) - 1 

        # Lists for input and target vectors/values
        session_x = [None]*(self.max_length-1)
        session_y = [None]*real_session_length

        # 0-real_session_length are inputs, 1-real_session_length+1 are targets
        for i in range(real_session_length):
            session_x[i] = self.one_hots[raw_data[i]]
            session_y[i] = raw_data[i+1]

        # Number of dummy vectors we need to add
        padding_length = self.max_length - len(raw_data)
        for i in range(padding_length):
            # Only pad input, since the RNN wont get the targets
            session_x[real_session_length+i] = self.dummy_vector

        return session_x, session_y, real_session_length
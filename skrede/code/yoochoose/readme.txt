The code in this folder is an attempt at creating an RNN model on the 
yoochoose dataset (http://recsys.yoochoose.net/), based on the paper 
"Session-based Recommendations with Recurrent Neural Networks" by 
Hidasi et al.

To run the code, you must download the dataset from the link, and set 
the base_path variable in the two code files to point to the downloaded 
dataset.

    ## PREPROCESSING ##
The preprocessing is done using Hidasis code which can be found here:
    https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/preprocess.py
    It takes a little while to run..
The steps of the preprocessing:
    - Discard sessions of length 1
    - Remove items that appear less than 5 times
    - Again discard sessions of length 1
    - Then the sessions are distributed in training and test set, based on time.
        Sessions from the subsequent day goes in the test set

    In the end we end up with these sets:
        Full train set
            Events: 31637239
            Sessions: 7966257
            Items: 37483
        Test set
            Events: 71222
            Sessions: 15324
            Items: 6751                    
        Train set
            Events: 31579006
            Sessions: 7953885
            Items: 37483
        Validation set
            Events: 58233
            Sessions: 12372
            Items: 6359

	In the prepocessed files there are 3 columns:
		sessionID | itemID | unix timestamp
	
	I then run a second preprocessing script (preprocess.py) on the preprocessed data to 
	get it in a format that is easier to work with. In this new format there is one line per
	session, and that line contains the clicks (item_id) in order, seperated by a space.
	TODO: Controll the assumption that Hidasi's preprocessing outputs the clicks within a session
	in order (by time).
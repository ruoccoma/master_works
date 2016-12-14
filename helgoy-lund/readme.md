# Prosjektoppgave: Multimodal deep learning

#### Preprocessing

##### Word embedding
To create new word embeddings edit `settings.py` and run:

`python models/word_embedding/word_embedding_main.py`

##### Image embeddings
To create new word embeddings run either:

`python models/image_embedding/vgg/vgg_19.py`
for VGG-19 or

`python models/image_embedding/inception/cnn_imagenet.py`
for Inception.

#### Training
To train a new model edit `main.py` and add new architectures to ARCHITECTURES array. Then run:
`python models/word2visualvec/main.py`

#### Evaluation
`python models/word2visualvec/main.py eval`

#### Random text-to-image retrieval
`python models/word2visualvec/main.py sample_image_query`

#### Requirements
The following python packages are required. Installed using "pip". 

- args==0.1.0
- Keras==1.1.1
- numpy==1.11.2
- Pillow==3.4.2
- protobuf==3.0.0
- pydot==1.2.3
- pyparsing==2.1.10
- PyYAML==3.12
- scikit-learn==0.18.1
- scipy==0.18.1
- six==1.10.0
- sklearn==0.0
- tensorflow==0.11.0
- Theano==0.8.2
- bson==0.4.3
- decorator==4.0.10
- funcsigs==1.0.2
- h5py==2.6.0
- nltk==3.2.1
- pymongo==3.3.0
- pytz==2016.7
- requests==2.11.1
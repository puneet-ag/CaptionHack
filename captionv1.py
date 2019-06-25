from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, CuDNNLSTM
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import numpy as np
from keras.preprocessing import text
from keras.utils import to_categorical


wandb.init()
config = wandb.config


train_txt = "../flickr/Flickr_8k.trainImages.txt"
val_txt = "../flickr/Flickr_8k.devImages.txt"
test_txt = "../flickr/Flickr_8k.testImages.txt"
token_txt = "../flickr/Flickr8k.token.txt" 
data_dir = "../flickr/datasets"


captions = [e.strip().split("\t") for e in open(token_txt).readlines()]
trainfiles = [e.strip() for e in open(train_txt).readlines()]
devfiles = [e.strip() for e in open(val_txt).readlines()]
testfiles = [e.strip() for e in open(test_txt).readlines()]

from collections import defaultdict
captiondict = defaultdict(list)
for pair in captions:
    image = pair[0].split('#')[0]
    caption = pair[1]
    captiondict[image].append(caption)
traindict = {}
devdict = {}
testdict = {}
for f in trainfiles:
    traindict[f] = captiondict[f]
for f in devfiles:
    devdict[f] = captiondict[f]
for f in testfiles:
    testdict[f] = captiondict[f]
    
tokenizer = text.Tokenizer()

captions_train = [captiondict[f] for f in trainfiles]
captions_valid = [captiondict[f] for f in devfiles]
captions_test = [captiondict[f] for f in testfiles]

from  itertools import chain
train_list = list(chain.from_iterable(traindict.values()))
validation_list = list(chain.from_iterable(devdict.values()))
test_list = list(chain.from_iterable(testdict.values()))


tokenizer.fit_on_texts(train_list)
X_train = tokenizer.texts_to_sequences(train_list)
X_valid = tokenizer.texts_to_sequences(validation_list)
X_test = tokenizer.texts_to_sequences(test_list)

config.num_classes = 2000
config.train_size = 50000
config.test_size = 10000
config.batch_size = 32
config.embedding_dims = 50
config.hidden_dims = 10
config.epochs = 10

word = []
next_word = []
for list_t in X_train:
    for i in range(1,len(list_t)):
        word.append(list_t[i-1])
        next_word.append(list_t[i])

        
word_val = []
next_word_val = []
for list_t in X_valid:
    for i in range(1,len(list_t)):
        word_val.append(list_t[i-1])
        next_word_val.append(list_t[i])
        
X_train_f= to_categorical(word[config.train_size], num_classes= config.num_classes)
y_train_f = to_categorical(next_word[config.train_size], num_classes = config.num_classes)
X_valid_f = to_categorical(word_val[config.test_size], num_classes= config.num_classes)
y_valid_f = to_categorical(next_word_val[config.test_size], num_classes= config.num_classes)

model = Sequential()
model.add(Embedding(config.num_classes, config.embedding_dims, input_length = config.train_size))
model.add(CuDNNLSTM(config.hidden_dims))
model.add(Dense(config.num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop')

print(model.summary())

model.fit(X_train_f, y_train_f,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_valid_f, y_valid_f), callbacks=[WandbCallback()])
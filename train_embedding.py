import os, random
import tensorflow as tf
import numpy as np

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils

import pickle

from utils import save_model_to_json

from preprocess import \
    load_dataframe, process_dataframe, build_tokenizer, to_array

train_path = './dataset/train.csv'
test_path = './dataset/test.csv'
train_df = load_dataframe(train_path)
test_df = load_dataframe(test_path)

train_df = process_dataframe(train_df)
test_df = process_dataframe(test_df)

tokenizer, vocab_size = build_tokenizer(train_df)

training_sentences, training_labels = to_array(train_df, tokenizer)
testing_sentences, testing_labels = to_array(test_df, tokenizer)

embedding_dim = 64
max_length = training_sentences.shape[1]
print(embedding_dim, max_length)

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 200, input_length=max_length),
    layers.Bidirectional(layers.LSTM(embedding_dim)),
    layers.Dense(1024, activation=activations.relu),
    layers.Dense(3, activation=activations.softmax)
])

optimizer = optimizers.SGD(lr=0.01)
model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

model.summary()

training_categorical = utils.to_categorical(training_labels, num_classes=3)
testing_categorical = utils.to_categorical(testing_labels, num_classes=3)
print(len(training_labels))
print(len(training_categorical))

history = model.fit(training_sentences, training_categorical, epochs=100,
                    validation_data=(testing_sentences, testing_categorical), verbose=2)
save_model_to_json(model, './results/neural_network_embedding')

with open('results/embedding.data', 'wb') as file:
    pickle.dump(history.history, file)

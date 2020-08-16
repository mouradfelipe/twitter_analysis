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
from tqdm import tqdm

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

word_length = 200
embedding_vector = {}
with open('./dataset/glove.twitter.27B.200d.txt') as file:
    for line in tqdm(file):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef

embedding_matrix = np.zeros((vocab_size, word_length))
for word, i in tqdm(tokenizer.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

training_sentences, training_labels = to_array(train_df, tokenizer)
testing_sentences, testing_labels = to_array(test_df, tokenizer)

embedding_dim = 64
max_length = training_sentences.shape[1]
print(embedding_dim, max_length)

model = tf.keras.Sequential([
    # (None, 35, 200)
    layers.Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False),
    # (None, 35, 1024)
    layers.Conv1D(filters=512, kernel_size=5, activation=activations.relu, padding='same'),
    # (None, 1, 1024)
    layers.MaxPooling1D(pool_size=5),
    # (None, 1024)
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
save_model_to_json(model, './results/neural_network_crnn')

with open('results/crnn.data', 'wb') as file:
    # store the data as binary data stream
    pickle.dump(history.history, file)

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils

import numpy as np
from tqdm import tqdm

from utils import save_model_to_json
from preprocess import \
    load_dataframe, process_dataframe, build_tokenizer, split_dataframe, to_array

train_path = './dataset/train.csv'
test_path = './dataset/test.csv'
train_df = load_dataframe(train_path)
test_df = load_dataframe(test_path)
#print(train_df.head())

train_df = process_dataframe(train_df)
test_df = process_dataframe(test_df)
#print(train_df.head())

# train_df, test_df = split_dataframe(df, 0.2)
tokenizer, vocab_size = build_tokenizer(train_df)

embedding_vector = {}
f = open('./dataset/glove.twitter.27B.200d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:], dtype='float32')
    embedding_vector[word] = coef

embedding_matrix = np.zeros((vocab_size, 200))
for word, i in tqdm(tokenizer.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

training_sentences, training_labels = to_array(train_df, tokenizer)
testing_sentences, testing_labels = to_array(test_df, tokenizer)

embedding_dim = 128
max_length = training_sentences.shape[1]
print(embedding_dim, max_length)

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=max_length, trainable=False),
    layers.Bidirectional(layers.LSTM(embedding_dim)),
    layers.Dense(1024, activation=activations.relu),
    layers.Dense(3, activation=activations.softmax)
])

optimizer = optimizers.SGD(lr=0.001, nesterov=True)
model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

model.summary()

training_categorical = utils.to_categorical(training_labels, num_classes=3)
testing_categorical = utils.to_categorical(testing_labels, num_classes=3)
print(len(training_labels))
print(len(training_categorical))

history = model.fit(training_sentences, training_categorical, epochs=1000, batch_size=1024,
                    validation_data=(testing_sentences, testing_categorical), verbose=2)


# MODEL_NAME = 'neutal_network'
# PATH_OF_MODEL = MODEL_NAME + '.json'
# if not isfile(PATH_OF_MODEL): 
save_model_to_json(model, 'neural_network_glove')


import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from utils import save_model_to_json
from os.path import isfile
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

training_sentences, training_labels = to_array(train_df, tokenizer)
testing_sentences, testing_labels = to_array(test_df, tokenizer)

embedding_dim = 64
max_length = training_sentences.shape[1]
print(embedding_dim, max_length)

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim),
    layers.Bidirectional(layers.LSTM(embedding_dim)),
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


# MODEL_NAME = 'neutal_network'
# PATH_OF_MODEL = MODEL_NAME + '.json'
# if not isfile(PATH_OF_MODEL): 
save_model_to_json(model, 'neural_network')


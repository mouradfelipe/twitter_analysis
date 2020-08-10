import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils


from preprocess import \
    load_dataframe, process_dataframe, build_tokenizer, split_dataframe, to_array

path = './dataset/train.csv'
df = load_dataframe(path)
print(df.head())

df = process_dataframe(df)
print(df.head())

train_df, test_df = split_dataframe(df, 0.2)
tokenizer, vocab_size = build_tokenizer(train_df)

training_sentences, training_labels = to_array(train_df, tokenizer)
testing_sentences, testing_labels = to_array(test_df, tokenizer)

embedding_dim = 64
max_length = training_sentences.shape[1]
print(embedding_dim, max_length)

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation=activations.relu),
    layers.Dense(3, activation=activations.softmax)
])

optimizer = optimizers.Adam(lr=0.1)
model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

model.summary()

training_labels = utils.to_categorical(training_labels, num_classes=3)
testing_labels = utils.to_categorical(testing_labels, num_classes=3)
history = model.fit(training_sentences, training_labels, epochs=100,
                    validation_data=(testing_sentences, testing_labels), verbose=2)


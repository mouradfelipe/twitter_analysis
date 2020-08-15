from tensorflow.keras import utils
import numpy as np

import pickle
from tqdm import tqdm

from utils import save_model_to_json

from preprocess import \
    load_dataframe, process_dataframe, build_tokenizer, to_array, load_emoji_dataframe
from text_interpreter_nn import TextInterpreterNN

train_path = './dataset/train.csv'
test_path = './dataset/test.csv'
train_df = load_dataframe(train_path)
test_df = load_dataframe(test_path)

train_df = process_dataframe(train_df)
test_df = process_dataframe(test_df)

train_df_emoji = load_emoji_dataframe(train_df)
test_df_emoji = load_emoji_dataframe(test_df)

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
input_size = training_sentences.shape[1]
input_emoji = len(train_df_emoji.columns.to_list())
print(input_size, input_emoji)

text_interpreter = TextInterpreterNN(input_size, vocab_size, embedding_dim, embedding_matrix)
text_interpreter.insert_emoji_feature(input_size=input_emoji)
model = text_interpreter.get_model()
model.summary()

training_categorical = utils.to_categorical(training_labels, num_classes=3)
testing_categorical = utils.to_categorical(testing_labels, num_classes=3)


history = model.fit([training_sentences, train_df_emoji.to_numpy()], training_categorical, epochs=100,
                    validation_data=([testing_sentences, test_df_emoji.to_numpy()], testing_categorical), verbose=2)
save_model_to_json(model, './results/neural_network_extraction')

with open('results/extraction.data', 'wb') as file:
    pickle.dump(history.history, file)

from tensorflow.keras import utils
import numpy as np
from utils import save_model_to_json
from os.path import isfile
from preprocess import \
    load_dataframe, process_dataframe, build_tokenizer, split_dataframe, to_array,load_emoji_dataframe

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

training_sentences, training_labels = to_array(train_df, tokenizer)
testing_sentences, testing_labels = to_array(test_df, tokenizer)

embedding_dim = 64
#max_length = training_sentences.shape[1]

input_size = 3
input_emoji = len(train_df_emoji.columns.to_list())

text_interpreter = TextInterpreterNN(input_size,vocab_size,embedding_dim)
text_interpreter.insert_emoji_feature(input_size = input_emoji)
model = text_interpreter.get_model()


training_categorical = utils.to_categorical(training_labels, num_classes=3)
testing_categorical = utils.to_categorical(testing_labels, num_classes=3)


history = model.fit([training_sentences,train_df_emoji.to_numpy()], training_categorical, epochs=100,
                    validation_data=([testing_sentences,test_df_emoji.to_numpy()], testing_categorical), verbose=2)


# MODEL_NAME = 'neutal_network'
# PATH_OF_MODEL = MODEL_NAME + '.json'
# if not isfile(PATH_OF_MODEL): 
save_model_to_json(model, 'neural_network')


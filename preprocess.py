import pandas as pd
import numpy as np
from utils import emoji_reader,get_emojis

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_dataframe(path):
    df = pd.read_csv(path)
    return df


def process_dataframe(df):
    # replacing sentiment column with numbers
    replace_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].replace(to_replace=replace_dict)

    # dropping unnecessary columns
    cols = [col for col in df.columns if col != 'text' and col != 'sentiment']
    df = df.drop(labels=cols, axis=1)

    # filtering empty texts
    df = df.dropna(axis='index')

    # other filters here

    return df

def load_emoji_dataframe(dataframe):

    df = pd.DataFrame()

    for emoji in get_emojis():
        df[emoji] = 0

    for _,row in dataframe.iterrows():
        dict = emoji_reader(row.get('text'))
        df = df.append(dict,ignore_index = True)
        
    return df



def split_dataframe(df, frac):
    test = df.sample(frac=frac)
    train = df.drop(test.index)
    return train, test


def build_tokenizer(df):
    oov_tok = "<OOV>"
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(df['text'])

    # print(tokenizer.word_index)
    vocab_size = len(tokenizer.word_index) + 1  # +1 because of oov_tok
    return tokenizer, vocab_size


def to_array(df, tokenizer):
    tokens = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(tokens, maxlen=32)

    sentences = np.array(padded)
    labels = np.array(df['sentiment'])
    return sentences, labels

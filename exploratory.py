import random

from preprocess import \
    load_dataframe, process_dataframe, build_tokenizer, split_dataframe, to_array


train_path = './dataset/train.csv'
test_path = './dataset/test.csv'

train_df = load_dataframe(train_path)
test_df = load_dataframe(test_path)

print('\nExemplos de treinamento')
for i in range(5):
    idx = random.randrange(0, len(train_df.index))
    print(train_df.iloc[23])

print('\nFrases de treinamento')
for line in random.sample(train_df['text'].tolist(), 5):
    print(line)

train_df = process_dataframe(train_df)
test_df = process_dataframe(test_df)

print('\nDataFrame de treinamento')
print(train_df.groupby('sentiment').count())

print('\nDataFrame de teste')
print(test_df.groupby('sentiment').count())

tokenizer, vocab_size = build_tokenizer(train_df)
keys = tokenizer.word_index.keys()

print('\nExemplos de palavras reconhecidas pelo Tokenizer')
print(random.sample(keys, 50))
print(tokenizer.word_index)

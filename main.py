from preprocess import \
    load_dataframe, load_emoji_dataframe,build_tokenizer,sample_text_to_array,classify_sentiment
from utils import load_model_from_json

path_to_trump_csv = './dataset/trump.csv'
path_to_biden_csv = './dataset/Biden.csv'


trump_tweets_df = load_dataframe(path_to_trump_csv)
tokenizer, _ = build_tokenizer(trump_tweets_df)
emojis_trump_tweets_df = load_emoji_dataframe(trump_tweets_df)
sample_trump = sample_text_to_array(trump_tweets_df,tokenizer)

# biden_tweets_df = load_dataframe(path_to_biden_csv)
# tokenizer, _ = build_tokenizer(biden_tweets_df)
# emojis_biden_tweets_df = load_emoji_dataframe(biden_tweets_df)
# sample_biden = sample_text_to_array(biden_tweets_df,tokenizer)


model_name = 'neural_network'
model = load_model_from_json(model_name)

predicted_labels = model.predict([sample_trump, emojis_trump_tweets_df.to_numpy()])
print(classify_sentiment(predicted_labels))

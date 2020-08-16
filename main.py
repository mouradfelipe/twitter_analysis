from preprocess import \
    load_dataframe, load_emoji_dataframe, build_tokenizer, sample_text_to_array, classify_sentiment, calculate_avgs
from utils import load_model_from_json,print_twitter_sentiment


path_to_trump_csv = 'dataset/trump.csv'
path_to_biden_csv = 'dataset/biden.csv'

trump_tweets_df = load_dataframe(path_to_trump_csv)
tokenizer, _ = build_tokenizer(trump_tweets_df)
emojis_trump_tweets_df = load_emoji_dataframe(trump_tweets_df)
sample_trump = sample_text_to_array(trump_tweets_df, tokenizer)

biden_tweets_df = load_dataframe(path_to_biden_csv)
tokenizer, _ = build_tokenizer(biden_tweets_df)
emojis_biden_tweets_df = load_emoji_dataframe(biden_tweets_df)
sample_biden = sample_text_to_array(biden_tweets_df,tokenizer)


model_name = 'results/lr01_neural_network_extraction'
model = load_model_from_json(model_name)

# predicted_labels = model.predict([sample_trump, emojis_trump_tweets_df.to_numpy()])
predicted_labels = model.predict([sample_biden, emojis_biden_tweets_df.to_numpy()])
sentiments = classify_sentiment(predicted_labels)
print_twitter_sentiment(calculate_avgs(sentiments),'Joe Biden')

predicted_labels = model.predict([sample_trump, emojis_trump_tweets_df.to_numpy()])
sentiments = classify_sentiment(predicted_labels)
print_twitter_sentiment(calculate_avgs(sentiments),'Donald Trump')

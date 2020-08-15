import tweepy as tw
import pandas as pd

with open('twitter-tokens.txt', 'r') as tfile:
    consumer_key = tfile.readline().strip('\n')
    consumer_secret = tfile.readline().strip('\n')
    access_token = tfile.readline().strip('\n')
    access_token_secret = tfile.readline().strip('\n')

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

candidate = "Biden"
# candidate = "Trump"
query_search = candidate + " -filter:retweets"
num_items = 10000

cursor_tweets = tw.Cursor(api.search, q=query_search, tweet_mode='extended').items(num_items)

tweets_dict = {"created_at": [], "full_text": []}
i = 1
for tweet in cursor_tweets:
    print(i)
    i += 1
    if tweet._json["lang"] == "en":
        for key in tweets_dict.keys():
            twvalue = tweet._json[key]
            tweets_dict[key].append(twvalue)

dfTweets = pd.DataFrame.from_dict(tweets_dict)
dfTweets.head()
dfTweets.to_csv(candidate + ".csv", index=False)

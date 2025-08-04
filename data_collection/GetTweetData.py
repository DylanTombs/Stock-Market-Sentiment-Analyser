import pandas as pd
import os
import tweepy

API_key = "" # Replace with your Twitter API key
API_key_secret = "" # Replace with your Twitter API key secret

Access_token = "" # Replace with your Twitter Access token
Access_token_secret = "" # Replace with your Twitter Access token secret

auth = tweepy.OAuth1UserHandler(API_key, API_key_secret, Access_token, Access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

def get_tweet_data(user, num_tweets, phrase = None):
    # Search query construction, dependent on whether a phrase is provided
    if phrase:
        search_query = f"from:{user}'{phrase}'-filter:retweets AND -filter:replies AND -filter:links"
        save_file = f"{user}_{phrase}_tweets.csv"
    else:
        search_query = f"'{user}'-filter:retweets AND -filter:replies AND -filter:links"
        save_file = f"{user}_tweets.csv"

    #Checking if the file/data already exists
    if os.path.exists(save_file):
        tweets = pd.read_csv(save_file, parse_dates=['Date'])
        return tweets

    num_tweets = min(num_tweets, 100)

    try:
        tweets = api.search_tweets(q=search_query, count=num_tweets, tweet_mode='extended', lang='en')

        # Preparing tweet data for DataFrame construction
        attributes = [[tweet.user.name, tweet.created_at, tweet.full_text, tweet.favorite_count, tweet.retweet_count] for tweet in tweets]
        columns = ["User", "Date", "Text", "Likes", "Retweets"]

        tweets = pd.DataFrame(attributes, columns=columns)
        tweets.to_csv(save_file, index=False)
        return tweets

    except Exception as e:
        print(f"Error fetching tweets for {user}: {e}")
        return pd.DataFrame()

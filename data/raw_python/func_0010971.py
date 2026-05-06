def get_tweepy_auth(twitter_api_key,
                    twitter_api_secret,
                    twitter_access_token,
                    twitter_access_token_secret):
    """Make a tweepy auth object"""
    auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
    auth.set_access_token(twitter_access_token, twitter_access_token_secret)
    return auth
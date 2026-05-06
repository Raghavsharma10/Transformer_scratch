def search_for_tweets_about(user_id, params):
    """ Search twitter API """
    url = "https://api.twitter.com/1.1/search/tweets.json"
    response = make_twitter_request(url, user_id, params)
    return process_tweets(response.json()["statuses"])
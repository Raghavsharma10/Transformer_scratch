def urlize_tweet(tweet):
    """ Turn #hashtag and @username in a text to Twitter hyperlinks,
        similar to the ``urlize()`` function in Django.
    """
    text = tweet.get('html', tweet['text'])
    for hash in tweet['entities']['hashtags']:
        text = text.replace('#%s' % hash['text'], TWITTER_HASHTAG_URL % (quote(hash['text'].encode("utf-8")), hash['text']))
    for mention in tweet['entities']['user_mentions']:
        text = text.replace('@%s' % mention['screen_name'], TWITTER_USERNAME_URL % (quote(mention['screen_name']), mention['screen_name']))
    tweet['html'] = text
    return tweet
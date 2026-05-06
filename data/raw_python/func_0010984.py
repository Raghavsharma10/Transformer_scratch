def expand_tweet_urls(tweet):
    """ Replace shortened URLs with long URLs in the twitter status, and add the "RT" flag.
        Should be used before urlize_tweet
    """
    if 'retweeted_status' in tweet:
        text = 'RT @{user}: {text}'.format(user=tweet['retweeted_status']['user']['screen_name'],
                                           text=tweet['retweeted_status']['text'])
        urls = tweet['retweeted_status']['entities']['urls']
    else:
        text = tweet['text']
        urls = tweet['entities']['urls']

    for url in urls:
        text = text.replace(url['url'], '<a href="%s">%s</a>' % (url['expanded_url'], url['display_url']))
    tweet['html'] = text
    return tweet
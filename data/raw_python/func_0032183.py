def urlize_tweet(tweet):
    """ Turn #hashtag and @username in a text to Twitter hyperlinks,
        similar to the ``urlize()`` function in Django.

        Replace shortened URLs with long URLs in the twitter status,
        and add the "RT" flag. Should be used before urlize_tweet
    """
    if tweet['retweeted']:
        text = 'RT {user}: {text}'.format(
            user=TWITTER_USERNAME_URL.format(screen_name=tweet['user']['screen_name']),
            text=tweet['text'])
    else:
        text = tweet['text']

    for hashtag in tweet['entities']['hashtags']:
        text = text.replace(
            '#%s' % hashtag['text'],
            TWITTER_HASHTAG_URL.format(hashtag=hashtag['text']))

    for mention in tweet['entities']['user_mentions']:
        text = text.replace(
            '@%s' % mention['screen_name'],
            TWITTER_USERNAME_URL.format(screen_name=mention['screen_name']))

    urls = tweet['entities']['urls']
    for url in urls:
        text = text.replace(
            url['url'], TWITTER_URL.format(
                url=url['expanded_url'], display_url=url['display_url']))

    if 'media' in tweet['entities']:
        for media in tweet['entities']['media']:
            text = text.replace(
                media['url'],
                TWITTER_MEDIA_URL.format(
                    url=media['expanded_url'],
                    display_url=media['display_url']))

    return mark_safe(text)
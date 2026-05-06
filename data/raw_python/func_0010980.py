def enrich(self, tweet):
        """ Apply the local presentation logic to the fetched data."""
        tweet = urlize_tweet(expand_tweet_urls(tweet))
        # parses created_at "Wed Aug 27 13:08:45 +0000 2008"

        if settings.USE_TZ:
            tweet['datetime'] = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y').replace(tzinfo=timezone.utc)
        else:
            tweet['datetime'] = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

        return tweet
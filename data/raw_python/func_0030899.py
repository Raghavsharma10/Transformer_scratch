def post_to_twitter(self, message=None):
        """Update twitter status, i.e., post a tweet"""

        consumer = oauth2.Consumer(key=conf.TWITTER_CONSUMER_KEY,
                                  secret=conf.TWITTER_CONSUMER_SECRET)
        token = oauth2.Token(key=conf.TWITTER_ACCESS_TOKEN, secret=conf.TWITTER_ACCESS_SECRET)
        client = SimpleTwitterClient(consumer=consumer, token=token)

        if not message:
            message = self.get_message()
        
        hash_tag = '#status'
        
        if conf.BASE_URL:
            permalink = urlparse.urljoin(conf.BASE_URL, reverse('overseer:event_short', args=[self.pk]))
            if len(message) + len(permalink) + len(hash_tag) > 138:
                message = '%s.. %s %s' % (message[:140-4-len(hash_tag)-len(permalink)], permalink, hash_tag)
            else:
                message = '%s %s %s' % (message, permalink, hash_tag) 
        else:
            if len(message) + len(hash_tag) > 139:
                message = '%s.. %s' % (message[:140-3-len(hash_tag)], hash_tag)
            else:
                message = '%s %s' % (message, hash_tag)
                
        return client.update_status(message)
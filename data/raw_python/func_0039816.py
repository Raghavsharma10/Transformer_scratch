def subscribe(self, sr):
        """Login required.  Send POST to subscribe to a subreddit.  If ``sr`` is the name of the subreddit, a GET request is sent to retrieve the full id of the subreddit, which is necessary for this API call.  Returns True or raises :class:`exceptions.UnexpectedResponse` if non-"truthy" value in response.
        
        URL: ``http://www.reddit.com/api/subscribe/``
        
        :param sr: full id of subreddit or name of subreddit (full id is preferred)
        """
        if not sr.startswith('t5_'):
            sr = self.subreddit(sr).name
        data = dict(action='sub', sr=sr)
        j = self.post('api', 'subscribe', data=data)
        return assert_truthy(j)
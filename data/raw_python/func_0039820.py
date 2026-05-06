def flaircsv(self, r, flair_csv):
        """Login required.  Bulk sets flair for users.  See https://github.com/reddit/reddit/wiki/API%3A-flaircsv/.  Returns response JSON content as dict.
        
        URL: ``http://www.reddit.com/api/flaircsv``
        
        :param r: name of subreddit
        :param flair_csv: csv string
        """
        # TODO: handle the response better than just returning
        data = dict(r=r, flair_csv=flair_csv)
        return self.post('api', 'flaircsv', data=data)
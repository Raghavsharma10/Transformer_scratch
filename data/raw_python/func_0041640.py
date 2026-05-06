def submit_link(self, title, url):
        """Submit link to this subreddit (POST).  Calls :meth:`narwal.Reddit.submit_link`.
        
        :param title: title of submission
        :param url: url submission links to
        """
        return self._reddit.submit_link(self.display_name, title, url)
def submit_text(self, title, text):
        """Submit self text submission to this subreddit (POST).  Calls :meth:`narwal.Reddit.submit_text`.
        
        :param title: title of submission
        :param text: self text
        """
        return self._reddit.submit_text(self.display_name, title, text)
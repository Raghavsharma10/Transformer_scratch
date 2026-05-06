def full_url(self):
        """Return the full reddit URL associated with the usernote.

        Arguments:
            subreddit: the subreddit name for the note (PRAW Subreddit object)
        """
        if self.link == '':
            return None
        else:
            return Note._expand_url(self.link, self.subreddit)
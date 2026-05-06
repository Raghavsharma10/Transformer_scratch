def manage_submissions(self):
        """
        If there are no or only one submissions left, get new submissions.
        This function manages URL creation and the specifics for front page
        or subreddit mode.
        """
        if not hasattr(self, 'submissions') or len(self.submissions) == 1:
            self.submissions = []
            if self.options['mode'] == 'front':
                # If there are no login details, the standard front
                # page will be displayed.
                if self.options['password'] and self.options['username']:
                    self.login()
                url = 'http://reddit.com/.json?sort={0}'.format(self.options['sort'])
                self.submissions = self.get_submissions(url)
            elif self.options['mode'] == 'subreddit':
                for subreddit in self.options['subreddits']:
                    url = 'http://reddit.com/r/{0}/.json?sort={1}'.format(
                        subreddit, self.options['limit'])
                    self.submissions += self.get_submissions(url)
        else:
            return
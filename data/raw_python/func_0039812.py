def submit_text(self, sr, title, text, follow=True):
        """Login required.  POSTs a text submission.  Returns :class:`things.Link` object if ``follow=True`` (default), or the string permalink of the new submission otherwise.
        
        Argument ``follow`` exists because reddit only returns the permalink after POSTing a submission.  In order to get detailed info on the new submission, we need to make another request.  If you don't want to make that additional request, set ``follow=False``.
        
        See https://github.com/reddit/reddit/wiki/API%3A-submit.
        
        URL: ``http://www.reddit.com/api/submit/``
        
        :param sr: name of subreddit to submit to
        :param title: title of submission
        :param text: submission self text
        :param follow: set to ``True`` to follow retrieved permalink to return detailed :class:`things.Link` object.  ``False`` to just return permalink.
        :type follow: bool
        """
        return self._submit(sr, title, 'self', text=text, follow=follow)
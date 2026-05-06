def flair(self, r, name, text, css_class):
        """Login required.  Sets flair for a user.  See https://github.com/reddit/reddit/wiki/API%3A-flair.  Returns True or raises :class:`exceptions.UnexpectedResponse` if non-"truthy" value in response.
        
        URL: ``http://www.reddit.com/api/flair``
        
        :param r: name of subreddit
        :param name: name of the user
        :param text: flair text to assign
        :param css_class: CSS class to assign to flair text
        """
        data = dict(r=r, name=name, text=text, css_class=css_class)
        j = self.post('api', 'flair', data=data)
        return assert_truthy(j)
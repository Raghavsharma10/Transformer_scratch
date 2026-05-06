def flair(self, name, text, css_class):
        """Sets flair for `user` in this subreddit (POST).  Calls :meth:`narwal.Reddit.flairlist`.
        
        :param name: name of the user
        :param text: flair text to assign
        :param css_class: CSS class to assign to flair text
        """
        return self._reddit.flair(self.display_name, name, text, css_class)
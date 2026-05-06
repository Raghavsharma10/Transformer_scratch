def unpublish(self):
        """
        Mark an episode as not published.
        """
        if self.published is True:
            self.published = False
        else:
            raise Warning(self.title + ' is already not published.')
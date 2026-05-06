def publish(self):
        """
        Mark an episode as published.
        """
        if self.published is False:
            self.published = True
        else:
            raise Warning(self.title + ' is already published.')
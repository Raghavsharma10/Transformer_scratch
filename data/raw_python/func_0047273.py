def title(self):
        """
        Title for the first header.
        """
        if self._title is None:
            self._title = self.get_title()
        return self._title
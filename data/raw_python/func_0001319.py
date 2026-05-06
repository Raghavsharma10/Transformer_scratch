def height(self):
        """Terminal height.
        """
        if self.interactive:
            if self._height is None:
                self._height = self.term.height
            return self._height
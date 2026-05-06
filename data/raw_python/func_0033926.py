def exists(self, index):
        """Checks whether :index: exists in the Model.

        :index: Index to look for.
        :returns: True if :index: exists in the Model, False otherwise.

        """
        data = self.data
        try:
            for c in self._split(index):
                i = int(c) - 1
                data = data[i][4]
        except Exception:
            return False
        return True
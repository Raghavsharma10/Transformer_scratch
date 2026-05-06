def drop(self):
        """
        Remove the database by deleting the JSON file.
        """
        import os

        if self.path:
            if os.path.exists(self.path):
                os.remove(self.path)
        else:
            # Clear the in-memory data if there is no file path
            self._data = {}
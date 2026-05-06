def delete(self, key):
        """
        Removes the specified key from the database.
        """
        obj = self._get_content()
        obj.pop(key, None)

        self.write_data(self.path, obj)
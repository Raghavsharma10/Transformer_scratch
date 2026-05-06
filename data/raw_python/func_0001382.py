def insert(self, index, value):
        """Insert an instance of User into the collection."""
        self.check(value)
        self._user_list.insert(index, value)
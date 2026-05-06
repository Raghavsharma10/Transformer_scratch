def remove(self, id):
        """Remove a object by id
            Args:
                id (int): Object's id should be deleted
            Returns:
                len(int): affected rows
        """
        before_len = len(self.model.db)
        self.model.db = [t for t in self.model.db if t["id"] != id]
        if not self._batch.enable.is_set():
            self.model.save_db()
        return before_len - len(self.model.db)
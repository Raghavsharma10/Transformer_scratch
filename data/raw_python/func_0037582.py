def removeAll(self):
        """Remove all objects
            Returns:
                len(int): affected rows
        """
        before_len = len(self.model.db)
        self.model.db = []
        if not self._batch.enable.is_set():
            self.model.save_db()
        return before_len - len(self.model.db)
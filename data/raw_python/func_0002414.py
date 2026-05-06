def force_delete(self):
        """
        Force a hard delete on a soft deleted model.
        """
        self._force_deleting = True

        self.delete()

        self._force_deleting = False
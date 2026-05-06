def save(self):
        """
        Save (modify) the storage to the API.
        Note: only size and title are updateable fields.
        """
        res = self.cloud_manager._modify_storage(self, self.size, self.title)
        self._reset(**res['storage'])
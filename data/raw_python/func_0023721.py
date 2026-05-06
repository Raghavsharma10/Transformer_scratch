def modify_storage(self, storage, size, title, backup_rule={}):
        """
        Modify a Storage object. Returns an object based on the API's response.
        """
        res = self._modify_storage(str(storage), size, title, backup_rule)
        return Storage(cloud_manager=self, **res['storage'])
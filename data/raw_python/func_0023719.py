def get_storage(self, storage):
        """
        Return a Storage object from the API.
        """
        res = self.get_request('/storage/' + str(storage))
        return Storage(cloud_manager=self, **res['storage'])
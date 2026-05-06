def load(self, id, *args, **kwargs):
        """
        loads a remote resource by id
        """
        self._pre_load(id, *args, **kwargs)
        response = self._load(id, *args, **kwargs)
        response = self._post_load(response, *args, **kwargs)
        return response
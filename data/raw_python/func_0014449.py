def create(self, name, numShards, params=None):
        """
        Create a new collection.
        """
        if params is None:
            params = {}
        params.update(
            name=name,
            numShards=numShards
        )
        return self.api('CREATE', params)
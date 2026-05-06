def put_collection(self, collection, body):
        """
        Updates an existing collection.

        The collection being updated *is* expected to include the id.
        """

        uri = self.uri + '/v1' + collection
        return self.service._put(uri, body)
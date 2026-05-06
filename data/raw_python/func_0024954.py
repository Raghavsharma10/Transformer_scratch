def delete_collection(self, collection):
        """
        Deletes an existing collection.

        The collection being updated *is* expected to include the id.
        """
        uri = str.join('/', [self.uri, collection])
        return self.service._delete(uri)
def get(self, workflow_id):
        """ Returns the document for the given workflow id.

        Args:
            workflow_id (str): The id of the document that represents a workflow run.

        Raises:
            DataStoreNotConnected: If the data store is not connected to the server.

        Returns:
            DataStoreDocument: The document for the given workflow id.
        """
        try:
            db = self._client[self.database]
            fs = GridFSProxy(GridFS(db.unproxied_object))
            return DataStoreDocument(db[WORKFLOW_DATA_COLLECTION_NAME], fs, workflow_id)

        except ConnectionFailure:
            raise DataStoreNotConnected()
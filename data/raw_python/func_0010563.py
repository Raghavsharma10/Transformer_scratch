def exists(self, workflow_id):
        """ Checks whether a document with the specified workflow id already exists.

        Args:
            workflow_id (str): The workflow id that should be checked.

        Raises:
            DataStoreNotConnected: If the data store is not connected to the server.

        Returns:
            bool: ``True`` if a document with the specified workflow id exists.
        """
        try:
            db = self._client[self.database]
            col = db[WORKFLOW_DATA_COLLECTION_NAME]
            return col.find_one({"_id": ObjectId(workflow_id)}) is not None

        except ConnectionFailure:
            raise DataStoreNotConnected()
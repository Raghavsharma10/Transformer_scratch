def add(self, payload=None):
        """ Adds a new document to the data store and returns its id.

        Args:
            payload (dict): Dictionary of initial data that should be stored
                in the new document in the meta section.

        Raises:
            DataStoreNotConnected: If the data store is not connected to the server.

        Returns:
            str: The id of the newly created document.
        """
        try:
            db = self._client[self.database]
            col = db[WORKFLOW_DATA_COLLECTION_NAME]
            return str(col.insert_one({
                DataStoreDocumentSection.Meta:
                    payload if isinstance(payload, dict) else {},
                DataStoreDocumentSection.Data: {}
            }).inserted_id)

        except ConnectionFailure:
            raise DataStoreNotConnected()
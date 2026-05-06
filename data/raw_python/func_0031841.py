def delete(self, database, key, callback=None):
        """
        Delete an item from the given database.

        :param database: The database from which to delete the value.
        :type database: .BlobDatabaseID
        :param key: The key to delete.
        :type key: uuid.UUID
        :param callback: A callback to be called on success or failure.
        """
        token = self._get_token()
        self._enqueue(self._PendingItem(token, BlobCommand(token=token, database=database,
                                                           content=DeleteCommand(key=key.bytes)),
                                        callback))
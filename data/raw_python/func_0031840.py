def insert(self, database, key, value, callback=None):
        """
        Insert an item into the given database.

        :param database: The database into which to insert the value.
        :type database: .BlobDatabaseID
        :param key: The key to insert.
        :type key: uuid.UUID
        :param value: The value to insert.
        :type value: bytes
        :param callback: A callback to be called on success or failure.
        """
        token = self._get_token()
        self._enqueue(self._PendingItem(token, BlobCommand(token=token, database=database,
                                                           content=InsertCommand(key=key.bytes, value=value)),
                                        callback))
def clear(self, database, callback=None):
        """
        Wipe the given database. This only affects items inserted remotely; items inserted on the watch
        (e.g. alarm clock timeline pins) are not removed.

        :param database: The database to wipe.
        :type database: .BlobDatabaseID
        :param callback: A callback to be called on success or failure.
        """
        token = self._get_token()
        self._enqueue(self._PendingItem(token, BlobCommand(token=token, database=database,
                                                           content=ClearCommand()),
                                        callback))
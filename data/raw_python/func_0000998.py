def open(self):
        """Opens the connection."""
        self._id = str(uuid.uuid4())
        self._client.open_connection(self._id, info=self._connection_args)
def connect(self, host=None, port=None, connect=False, **kwargs):
        """ Explicitly creates the MongoClient; this method must be used
            in order to specify a non-default host or port to the MongoClient.
            Takes arguments identical to MongoClient.__init__"""
        try:
            self.__connection = MongoClient(host=host, port=port, connect=connect, **kwargs)
        except (AutoReconnect, ConnectionFailure, ServerSelectionTimeoutError):
            raise DatabaseIsDownError("No mongod process is running.")
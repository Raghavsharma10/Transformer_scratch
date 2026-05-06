def drop_connection(self, name, database=None):
        """
        Force server to close current client subscription connection to the server
        @param str name: The name of the subscription
        @param str database: The name of the database
        """
        request_executor = self._store.get_request_executor(database)
        command = DropSubscriptionConnectionCommand(name)
        request_executor.execute(command)
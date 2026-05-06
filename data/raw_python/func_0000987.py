def close_connection(self, connection_id):
        """Closes a connection.

        :param connection_id:
            ID of the connection to close.
        """
        request = requests_pb2.CloseConnectionRequest()
        request.connection_id = connection_id
        self._apply(request)
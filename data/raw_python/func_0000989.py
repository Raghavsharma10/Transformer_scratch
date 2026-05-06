def close_statement(self, connection_id, statement_id):
        """Closes a statement.

        :param connection_id:
            ID of the current connection.

        :param statement_id:
            ID of the statement to close.
        """
        request = requests_pb2.CloseStatementRequest()
        request.connection_id = connection_id
        request.statement_id = statement_id

        self._apply(request)
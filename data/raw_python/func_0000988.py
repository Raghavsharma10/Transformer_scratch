def create_statement(self, connection_id):
        """Creates a new statement.

        :param connection_id:
            ID of the current connection.

        :returns:
            New statement ID.
        """
        request = requests_pb2.CreateStatementRequest()
        request.connection_id = connection_id

        response_data = self._apply(request)
        response = responses_pb2.CreateStatementResponse()
        response.ParseFromString(response_data)
        return response.statement_id
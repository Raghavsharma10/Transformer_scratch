def prepare(self, connection_id, sql, max_rows_total=None):
        """Prepares a statement.

        :param connection_id:
            ID of the current connection.

        :param sql:
            SQL query.

        :param max_rows_total:
            The maximum number of rows that will be allowed for this query.

        :returns:
            Signature of the prepared statement.
        """
        request = requests_pb2.PrepareRequest()
        request.connection_id = connection_id
        request.sql = sql
        if max_rows_total is not None:
            request.max_rows_total = max_rows_total

        response_data = self._apply(request)
        response = responses_pb2.PrepareResponse()
        response.ParseFromString(response_data)
        return response.statement
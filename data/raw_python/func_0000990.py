def prepare_and_execute(self, connection_id, statement_id, sql, max_rows_total=None, first_frame_max_size=None):
        """Prepares and immediately executes a statement.

        :param connection_id:
            ID of the current connection.

        :param statement_id:
            ID of the statement to prepare.

        :param sql:
            SQL query.

        :param max_rows_total:
            The maximum number of rows that will be allowed for this query.

        :param first_frame_max_size:
            The maximum number of rows that will be returned in the first Frame returned for this query.

        :returns:
            Result set with the signature of the prepared statement and the first frame data.
        """
        request = requests_pb2.PrepareAndExecuteRequest()
        request.connection_id = connection_id
        request.statement_id = statement_id
        request.sql = sql
        if max_rows_total is not None:
            request.max_rows_total = max_rows_total
        if first_frame_max_size is not None:
            request.first_frame_max_size = first_frame_max_size

        response_data = self._apply(request, 'ExecuteResponse')
        response = responses_pb2.ExecuteResponse()
        response.ParseFromString(response_data)
        return response.results
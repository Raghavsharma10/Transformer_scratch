def execute(self, connection_id, statement_id, signature, parameter_values=None, first_frame_max_size=None):
        """Returns a frame of rows.

        The frame describes whether there may be another frame. If there is not
        another frame, the current iteration is done when we have finished the
        rows in the this frame.

        :param connection_id:
            ID of the current connection.

        :param statement_id:
            ID of the statement to fetch rows from.

        :param signature:
            common_pb2.Signature object

        :param parameter_values:
            A list of parameter values, if statement is to be executed; otherwise ``None``.

        :param first_frame_max_size:
            The maximum number of rows that will be returned in the first Frame returned for this query.

        :returns:
            Frame data, or ``None`` if there are no more.
        """
        request = requests_pb2.ExecuteRequest()
        request.statementHandle.id = statement_id
        request.statementHandle.connection_id = connection_id
        request.statementHandle.signature.CopyFrom(signature)
        if parameter_values is not None:
            request.parameter_values.extend(parameter_values)
            request.has_parameter_values = True
        if first_frame_max_size is not None:
            request.deprecated_first_frame_max_size = first_frame_max_size
            request.first_frame_max_size = first_frame_max_size

        response_data = self._apply(request)
        response = responses_pb2.ExecuteResponse()
        response.ParseFromString(response_data)
        return response.results
def fetch(self, connection_id, statement_id, offset=0, frame_max_size=None):
        """Returns a frame of rows.

        The frame describes whether there may be another frame. If there is not
        another frame, the current iteration is done when we have finished the
        rows in the this frame.

        :param connection_id:
            ID of the current connection.

        :param statement_id:
            ID of the statement to fetch rows from.

        :param offset:
            Zero-based offset of first row in the requested frame.

        :param frame_max_size:
            Maximum number of rows to return; negative means no limit.

        :returns:
            Frame data, or ``None`` if there are no more.
        """
        request = requests_pb2.FetchRequest()
        request.connection_id = connection_id
        request.statement_id = statement_id
        request.offset = offset
        if frame_max_size is not None:
            request.frame_max_size = frame_max_size

        response_data = self._apply(request)
        response = responses_pb2.FetchResponse()
        response.ParseFromString(response_data)
        return response.frame
def open_connection(self, connection_id, info=None):
        """Opens a new connection.

        :param connection_id:
            ID of the connection to open.
        """
        request = requests_pb2.OpenConnectionRequest()
        request.connection_id = connection_id
        if info is not None:
            # Info is a list of repeated pairs, setting a dict directly fails
            for k, v in info.items():
                request.info[k] = v

        response_data = self._apply(request)
        response = responses_pb2.OpenConnectionResponse()
        response.ParseFromString(response_data)
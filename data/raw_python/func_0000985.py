def connection_sync(self, connection_id, connProps=None):
        """Synchronizes connection properties with the server.

        :param connection_id:
            ID of the current connection.

        :param connProps:
            Dictionary with the properties that should be changed.

        :returns:
            A ``common_pb2.ConnectionProperties`` object.
        """
        if connProps is None:
            connProps = {}

        request = requests_pb2.ConnectionSyncRequest()
        request.connection_id = connection_id
        request.conn_props.auto_commit = connProps.get('autoCommit', False)
        request.conn_props.has_auto_commit = True
        request.conn_props.read_only = connProps.get('readOnly', False)
        request.conn_props.has_read_only = True
        request.conn_props.transaction_isolation = connProps.get('transactionIsolation', 0)
        request.conn_props.catalog = connProps.get('catalog', '')
        request.conn_props.schema = connProps.get('schema', '')

        response_data = self._apply(request)
        response = responses_pb2.ConnectionSyncResponse()
        response.ParseFromString(response_data)
        return response.conn_props
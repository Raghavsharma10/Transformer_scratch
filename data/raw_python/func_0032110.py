def query(self, transport, protocol, *data):
        """Generates and sends a query message unit.

        :param transport: An object implementing the `.Transport` interface.
            It is used by the protocol to send the message and receive the
            response.
        :param protocol: An object implementing the `.Protocol` interface.
        :param data: The program data.

        :raises AttributeError: if the command is not queryable.

        """
        if not self._query:
            raise AttributeError('Command is not queryable')
        if self.protocol:
            protocol = self.protocol
        if self._query.data_type:
            data = _dump(self._query.data_type, data)
        else:
            # TODO We silently ignore possible data
            data = ()
        if isinstance(transport, SimulatedTransport):
            response = self.simulate_query(data)
        else:
            response = protocol.query(transport, self._query.header, *data)
        response = _load(self._query.response_type, response)

        # Return single value if parsed_data is 1-tuple.
        return response[0] if len(response) == 1 else response
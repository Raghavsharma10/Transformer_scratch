def write(self, transport, protocol, *data):
        """Generates and sends a command message unit.

        :param transport: An object implementing the `.Transport` interface.
            It is used by the protocol to send the message.
        :param protocol: An object implementing the `.Protocol` interface.
        :param data: The program data.

        :raises AttributeError: if the command is not writable.

        """
        if not self._write:
            raise AttributeError('Command is not writeable')
        if self.protocol:
            protocol = self.protocol
        if self._write.data_type:
            data = _dump(self._write.data_type, data)
        else:
            # TODO We silently ignore possible data
            data = ()
        if isinstance(transport, SimulatedTransport):
            self.simulate_write(data)
        else:
            protocol.write(transport, self._write.header, *data)
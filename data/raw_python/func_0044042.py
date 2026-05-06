def send(self, content):
        """Sends a JavaScript command to PS

        :param content: Script content
        :type content: str
        :yields: :class:`.Message`
        """
        LOGGER.debug('Sending: %s', content)
        all_bytes = struct.pack('>i', Connection.PROTOCOL_VERSION)
        all_bytes += struct.pack('>i', self._id)
        all_bytes += struct.pack('>i', 2)
        self._id += 1
        for char in content:
            all_bytes += struct.pack('>c', char.encode('utf8'))

        encrypted_bytes = self._crypt.encrypt(all_bytes)

        message_length = Connection.COMM_LENGTH + len(encrypted_bytes)

        self._socket.send(struct.pack('>i', message_length))
        self._socket.send(struct.pack('>i', Connection.NO_COMM_ERROR))
        self._socket.send(encrypted_bytes)
        LOGGER.debug('Sent')

        message = self.recv()
        while message is None:
            message = self.recv()
            yield message

        yield message
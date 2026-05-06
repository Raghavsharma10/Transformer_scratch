def _get_response(self):
        """
        Get memcached response from socket.

        :return: A tuple with binary values from memcached.
        :rtype: tuple
        """
        try:
            self._open_connection()
            if self.connection is None:
                # The connection wasn't opened, which means we're deferring a reconnection attempt.
                # Raise a socket.error, so we'll return the same server_disconnected message as we
                # do below.
                raise socket.error('Delaying reconnection attempt')

            header = self._read_socket(self.HEADER_SIZE)
            (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque,
             cas) = struct.unpack(self.HEADER_STRUCT, header)

            assert magic == self.MAGIC['response']

            extra_content = None
            if bodylen:
                extra_content = self._read_socket(bodylen)

            return (magic, opcode, keylen, extlen, datatype, status, bodylen,
                    opaque, cas, extra_content)
        except socket.error as e:
            self._connection_error(e)

            # (magic, opcode, keylen, extlen, datatype, status, bodylen, opaque, cas, extra_content)
            message = str(e)
            return (self.MAGIC['response'], -1, 0, 0, 0, self.STATUS['server_disconnected'], 0, 0, 0, message)
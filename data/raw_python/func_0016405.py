def _send_msg(self, header, payload):
        """send message to server"""

        if self.verbose:
            print('->', repr(header))
            print('..', repr(payload))
        assert header.payload == len(payload)
        try:
            sent = self.socket.send(header + payload)
        except IOError as err:
            raise ConnError(*err.args)

        # FIXME FIXME FIXME:
        # investigate under which situations socket.send should be retried
        # instead of aborted.
        # FIXME FIXME FIXME
        if sent < len(header + payload):
            raise ShortWrite(sent, len(header + payload))
        assert sent == len(header + payload), sent
def shutdown(self):
        """shutdown connection"""

        if self.verbose:
            print(self.socket.getsockname(), 'xx', self.peername)

        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except IOError as err:
            assert err.errno is _ENOTCONN, "unexpected IOError: %s" % err
            # remote peer has already closed the connection,
            # just ignore the exceeption
            pass
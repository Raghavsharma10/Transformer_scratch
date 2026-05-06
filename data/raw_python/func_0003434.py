def read(self, len=1024, buffer=None):
        """Read data from connection

        Read up to len bytes and return them.
        Arguments:
        len -- maximum number of bytes to read

        Return value:
        string containing read bytes
        """

        try:
            return self._wrap_socket_library_call(
                lambda: SSL_read(self._ssl.value, len, buffer), ERR_READ_TIMEOUT)
        except openssl_error() as err:
            if err.ssl_error == SSL_ERROR_SYSCALL and err.result == -1:
                raise_ssl_error(ERR_PORT_UNREACHABLE, err)
            raise
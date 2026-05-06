def write(self, data):
        """Write data to connection

        Write data as string of bytes.

        Arguments:
        data -- buffer containing data to be written

        Return value:
        number of bytes actually transmitted
        """

        try:
            ret = self._wrap_socket_library_call(
                lambda: SSL_write(self._ssl.value, data), ERR_WRITE_TIMEOUT)
        except openssl_error() as err:
            if err.ssl_error == SSL_ERROR_SYSCALL and err.result == -1:
                raise_ssl_error(ERR_PORT_UNREACHABLE, err)
            raise
        if ret:
            self._handshake_done = True
        return ret
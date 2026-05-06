def write(self, data):
        """Write DATA to the underlying SSL channel.  Returns
        number of bytes of DATA actually transmitted."""
        while True:
            try:
                return self._sslobj.write(data)
            except SSLError:
                ex = sys.exc_info()[1]
                if ex.args[0] == SSL_ERROR_WANT_READ:
                    if self.timeout == 0.0:
                        raise
                    six.exc_clear()
                    self._io.wait_read(timeout=self.timeout, timeout_exc=_SSLErrorWriteTimeout)
                elif ex.args[0] == SSL_ERROR_WANT_WRITE:
                    if self.timeout == 0.0:
                        raise
                    six.exc_clear()
                    self._io.wait_write(timeout=self.timeout, timeout_exc=_SSLErrorWriteTimeout)
                else:
                    raise
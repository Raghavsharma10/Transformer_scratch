def read(self, len=1024):
        """Read up to LEN bytes and return them.
        Return zero-length string on EOF."""
        while True:
            try:
                return self._sslobj.read(len)
            except SSLError:
                ex = sys.exc_info()[1]
                if ex.args[0] == SSL_ERROR_EOF and self.suppress_ragged_eofs:
                    return b''
                elif ex.args[0] == SSL_ERROR_WANT_READ:
                    if self.timeout == 0.0:
                        raise
                    six.exc_clear()
                    self._io.wait_read(timeout=self.timeout, timeout_exc=_SSLErrorReadTimeout)
                elif ex.args[0] == SSL_ERROR_WANT_WRITE:
                    if self.timeout == 0.0:
                        raise
                    six.exc_clear()
                    self._io.wait_write(timeout=self.timeout, timeout_exc=_SSLErrorReadTimeout)
                else:
                    raise
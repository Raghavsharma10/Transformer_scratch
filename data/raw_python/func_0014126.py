def read(self, n):
    """ Consume `n` characters from the stream. """
    while len(self.buf) < n:
      chunk = self.f.recv(4096)
      if not chunk:
        raise EndOfStreamError()
      self.buf += chunk
    res, self.buf = self.buf[:n], self.buf[n:]
    return res
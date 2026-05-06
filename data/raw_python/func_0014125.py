def read_line(self):
    """ Consume one line from the stream. """
    while True:
      newline_idx = self.buf.find(b"\n")
      if newline_idx >= 0:
        res = self.buf[:newline_idx]
        self.buf = self.buf[newline_idx + 1:]
        return res
      chunk = self.f.recv(4096)
      if not chunk:
        raise EndOfStreamError()
      self.buf += chunk
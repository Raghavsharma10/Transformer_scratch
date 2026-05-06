def set_header(self, key, value):
    """ Sets a HTTP header for future requests. """
    self.conn.issue_command("Header", _normalize_header(key), value)
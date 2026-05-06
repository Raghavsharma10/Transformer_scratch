def headers(self):
    """ Returns a list of the last HTTP response headers.
    Header keys are normalized to capitalized form, as in `User-Agent`.
    """
    headers = self.conn.issue_command("Headers")
    res = []
    for header in headers.split("\r"):
      key, value = header.split(": ", 1)
      for line in value.split("\n"):
        res.append((_normalize_header(key), line))
    return res
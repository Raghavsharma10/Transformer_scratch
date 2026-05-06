def issue_command(self, cmd, *args):
    """ Sends and receives a message to/from the server """
    self._writeline(cmd)
    self._writeline(str(len(args)))
    for arg in args:
      arg = str(arg)
      self._writeline(str(len(arg)))
      self._sock.sendall(arg.encode("utf-8"))

    return self._read_response()
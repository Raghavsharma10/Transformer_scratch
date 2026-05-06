def simple_response(self, status, msg=""):
    """Return a operation for writing simple response back to the client."""
    status = str(status)
    buf = ["%s %s\r\n" % (self.environ['ACTUAL_SERVER_PROTOCOL'], status),
         "Content-Length: %s\r\n" % len(msg),
         "Content-Type: text/plain\r\n"]

    if status[:3] == "413" and self.response_protocol == 'HTTP/1.1':
      # Request Entity Too Large
      self.close_connection = True
      buf.append("Connection: close\r\n")

    buf.append("\r\n")
    if msg:
      buf.append(msg)
    return sockets.SendAll(self.conn, "".join(buf))
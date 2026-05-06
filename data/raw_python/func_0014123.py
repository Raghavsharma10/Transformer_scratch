def set_proxy(self, host     = "localhost",
                      port     = 0,
                      user     = "",
                      password = ""):
    """ Sets a custom HTTP proxy to use for future requests. """
    self.conn.issue_command("SetProxy", host, port, user, password)
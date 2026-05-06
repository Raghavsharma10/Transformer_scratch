def set_timeout(self, timeout):
        """ Set Screen Timeout Duration """

        if timeout > 0:
            self.timeout = timeout
            self.server.request("screen_set %s timeout %i" % (self.ref, (self.timeout * 8)))
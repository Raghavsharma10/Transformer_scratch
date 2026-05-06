def set_heartbeat(self, state):
        """ Set Screen Heartbeat Display Mode """

        if state in ["on", "off", "open"]:
            self.heartbeat = state
            self.server.request("screen_set %s heartbeat %s" % (self.ref, self.heartbeat))
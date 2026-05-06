def set_priority(self, priority):
        """ Set Screen Priority Class """

        if priority in ["hidden", "background", "info", "foreground", "alert", "input"]:
            self.priority = priority
            self.server.request("screen_set %s priority %s" % (self.ref, self.priority))
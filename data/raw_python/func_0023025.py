def set_cursor(self, cursor):
        """ Set Screen Cursor Mode """

        if cursor in ["on", "off", "under", "block"]:
            self.cursor = cursor
            self.server.request("screen_set %s cursor %s" % (self.ref, self.cursor))
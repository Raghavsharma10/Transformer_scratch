def set_width(self, width):
        """ Set Screen Width """

        if width > 0 and width <= self.server.server_info.get("screen_width"):
            self.width = width
            self.server.request("screen_set %s wid %i" % (self.ref, self.width))
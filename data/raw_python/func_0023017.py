def set_height(self, height):
        """ Set Screen Height """

        if height > 0 and height <= self.server.server_info.get("screen_height"):
            self.height = height
            self.server.request("screen_set %s hgt %i" % (self.ref, self.height))
def set_name(self, name):
        """ Set Screen Name """

        self.name = name
        self.server.request("screen_set %s name %s" % (self.ref, self.name))
def set_backlight(self, state):
        """ Set Screen Backlight Mode """

        if state in ["on", "off", "toggle", "open", "blink", "flash"]:
            self.backlight = state
            self.server.request("screen_set %s backlight %s" % (self.ref, self.backlight))
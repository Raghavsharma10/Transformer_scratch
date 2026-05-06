def set_cursor_y(self, y):
        """ Set Screen Cursor Y Position """

        if y >= 0 and y <= self.server.server_info.get("screen_height"):
            self.cursor_y = y
            self.server.request("screen_set %s cursor_y %i" % (self.ref, self.cursor_y))
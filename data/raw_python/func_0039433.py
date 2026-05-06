def set_title(self, msg):
        """ Set first header line text """
        self.s.move(0, 0)
        self.overwrite_line(msg, curses.A_REVERSE)
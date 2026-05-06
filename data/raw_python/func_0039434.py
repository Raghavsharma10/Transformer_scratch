def set_header(self, msg):
        """ Set second head line text """
        self.s.move(1, 0)
        self.overwrite_line(msg, attr=curses.A_NORMAL)
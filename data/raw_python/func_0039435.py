def set_footer(self, msg, reverse=True):
        """ Set first footer line text """
        self.s.move(self.max_y-1, 0)
        if reverse:
            self.overwrite_line(msg, attr=curses.A_REVERSE)
        else:
            self.overwrite_line(msg, attr=curses.A_NORMAL)
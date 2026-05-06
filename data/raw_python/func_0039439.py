def redraw_current_line(self):
        """ Redraw the highlighted line """
        if self.no_streams:
            return
        row = self.pads[self.current_pad].getyx()[0]
        s = self.filtered_streams[row]
        pad = self.pads['streams']
        pad.move(row, 0)
        pad.clrtoeol()
        pad.addstr(row, 0, self.format_stream_line(s), curses.A_REVERSE)
        pad.chgat(curses.A_REVERSE)
        pad.move(row, 0)
        self.refresh_current_pad()
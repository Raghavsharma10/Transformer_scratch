def init_streams_pad(self, start_row=0):
        """ Create a curses pad and populate it with a line by stream """
        y = 0
        pad = curses.newpad(max(1,len(self.filtered_streams)), self.pad_w)
        pad.keypad(1)
        for s in self.filtered_streams:
            pad.addstr(y, 0, self.format_stream_line(s))
            y+=1
        self.offsets['streams'] = 0
        pad.move(start_row, 0)
        if not self.no_stream_shown:
            pad.chgat(curses.A_REVERSE)
        self.pads['streams'] = pad
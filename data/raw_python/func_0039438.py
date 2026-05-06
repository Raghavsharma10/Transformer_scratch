def move(self, direction, absolute=False, pad_name=None, refresh=True):
        """ Scroll the current pad

        direction : (int)  move by one in the given direction
                           -1 is up, 1 is down. If absolute is True,
                           go to position direction.
                           Behaviour is affected by cursor_line and scroll_only below
        absolute  : (bool)
        """

        # pad in this lists have the current line highlighted
        cursor_line = [ 'streams' ]

        # pads in this list will be moved screen-wise as opposed to line-wise
        # if absolute is set, will go all the way top or all the way down depending
        # on direction
        scroll_only = [ 'help' ]

        if not pad_name:
            pad_name = self.current_pad
        pad = self.pads[pad_name]
        if pad_name == 'streams' and self.no_streams:
            return
        (row, col) = pad.getyx()
        new_row    = row
        offset = self.offsets[pad_name]
        new_offset = offset
        if pad_name in scroll_only:
            if absolute:
                if direction > 0:
                    new_offset = pad.getmaxyx()[0] - self.pad_h + 1
                else:
                    new_offset = 0
            else:
                if direction > 0:
                    new_offset = min(pad.getmaxyx()[0] - self.pad_h + 1, offset + self.pad_h)
                elif offset > 0:
                    new_offset = max(0, offset - self.pad_h)
        else:
            if absolute and direction >= 0 and direction < pad.getmaxyx()[0]:
                if direction < offset:
                    new_offset = direction
                elif direction > offset + self.pad_h - 2:
                    new_offset = direction - self.pad_h + 2
                new_row = direction
            else:
                if direction == -1 and row > 0:
                    if row == offset:
                        new_offset -= 1
                    new_row = row-1
                elif direction == 1 and row < len(self.filtered_streams)-1:
                    if row == offset + self.pad_h - 2:
                        new_offset += 1
                    new_row = row+1
        if pad_name in cursor_line:
            pad.move(row, 0)
            pad.chgat(curses.A_NORMAL)
        self.offsets[pad_name] = new_offset
        pad.move(new_row, 0)
        if pad_name in cursor_line:
            pad.chgat(curses.A_REVERSE)
        if pad_name == 'streams':
            self.redraw_stream_footer()
        if refresh:
            self.refresh_current_pad()
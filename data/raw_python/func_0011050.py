def render_to_terminal(self, array, cursor_pos=(0, 0)):
        """Renders array to terminal, returns the number of lines scrolled offscreen

        Returns:
            Number of times scrolled

        Args:
          array (FSArray): Grid of styled characters to be rendered.

            If array received is of width too small, render it anyway

            if array received is of width too large, render it anyway

            if array received is of height too small, render it anyway

            if array received is of height too large, render it, scroll down,
            and render the rest of it, then return how much we scrolled down

        """
        for_stdout = self.fmtstr_to_stdout_xform()
        # caching of write and tc (avoiding the self. lookups etc) made
        # no significant performance difference here
        if not self.hide_cursor:
            self.write(self.t.hide_cursor)

        # TODO race condition here?
        height, width = self.t.height, self.t.width
        if (height != self._last_rendered_height or
                width != self._last_rendered_width):
            self.on_terminal_size_change(height, width)

        current_lines_by_row = {}
        rows_for_use = list(range(self.top_usable_row, height))

        # rows which we have content for and don't require scrolling
        # TODO rename shared
        shared = min(len(array), len(rows_for_use))
        for row, line in zip(rows_for_use[:shared], array[:shared]):
            current_lines_by_row[row] = line
            if line == self._last_lines_by_row.get(row, None):
                continue
            self.write(self.t.move(row, 0))
            self.write(for_stdout(line))
            if len(line) < width:
                self.write(self.t.clear_eol)

        # rows already on screen that we don't have content for
        rest_of_lines = array[shared:]
        rest_of_rows = rows_for_use[shared:]
        for row in rest_of_rows:  # if array too small
            if self._last_lines_by_row and row not in self._last_lines_by_row:
                continue
            self.write(self.t.move(row, 0))
            self.write(self.t.clear_eol)
            # TODO probably not necessary - is first char cleared?
            self.write(self.t.clear_bol)
            current_lines_by_row[row] = None

        # lines for which we need to scroll down to render
        offscreen_scrolls = 0
        for line in rest_of_lines:  # if array too big
            self.scroll_down()
            if self.top_usable_row > 0:
                self.top_usable_row -= 1
            else:
                offscreen_scrolls += 1
            current_lines_by_row = dict(
                (k - 1, v) for k, v in current_lines_by_row.items()
            )
            logger.debug('new top_usable_row: %d' % self.top_usable_row)
            # since scrolling moves the cursor
            self.write(self.t.move(height - 1, 0))
            self.write(for_stdout(line))
            current_lines_by_row[height - 1] = line

        logger.debug(
            'lines in last lines by row: %r' % self._last_lines_by_row.keys()
        )
        logger.debug(
            'lines in current lines by row: %r' % current_lines_by_row.keys()
        )
        self._last_cursor_row = max(
            0, cursor_pos[0] - offscreen_scrolls + self.top_usable_row
        )
        self._last_cursor_column = cursor_pos[1]
        self.write(
            self.t.move(self._last_cursor_row, self._last_cursor_column)
        )
        self._last_lines_by_row = current_lines_by_row
        if not self.hide_cursor:
            self.write(self.t.normal_cursor)
        return offscreen_scrolls
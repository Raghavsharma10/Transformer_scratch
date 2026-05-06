def render_to_terminal(self, array, cursor_pos=(0, 0)):
        """Renders array to terminal and places (0-indexed) cursor

        Args:
            array (FSArray): Grid of styled characters to be rendered.

        * If array received is of width too small, render it anyway
        * If array received is of width too large,
        * render the renderable portion
        * If array received is of height too small, render it anyway
        * If array received is of height too large,
        * render the renderable portion (no scroll)
        """
        # TODO there's a race condition here - these height and widths are
        # super fresh - they might change between the array being constructed
        # and rendered
        # Maybe the right behavior is to throw away the render
        # in the signal handler?
        height, width = self.height, self.width

        for_stdout = self.fmtstr_to_stdout_xform()
        if not self.hide_cursor:
            self.write(self.t.hide_cursor)
        if (height != self._last_rendered_height or
                width != self._last_rendered_width):
            self.on_terminal_size_change(height, width)

        current_lines_by_row = {}
        rows = list(range(height))
        rows_for_use = rows[:len(array)]
        rest_of_rows = rows[len(array):]

        # rows which we have content for and don't require scrolling
        for row, line in zip(rows_for_use, array):
            current_lines_by_row[row] = line
            if line == self._last_lines_by_row.get(row, None):
                continue
            self.write(self.t.move(row, 0))
            self.write(for_stdout(line))
            if len(line) < width:
                self.write(self.t.clear_eol)

        # rows onscreen that we don't have content for
        for row in rest_of_rows:
            if self._last_lines_by_row and row not in self._last_lines_by_row:
                continue
            self.write(self.t.move(row, 0))
            self.write(self.t.clear_eol)
            self.write(self.t.clear_bol)
            current_lines_by_row[row] = None

        logger.debug(
            'lines in last lines by row: %r' % self._last_lines_by_row.keys()
        )
        logger.debug(
            'lines in current lines by row: %r' % current_lines_by_row.keys()
        )
        self.write(self.t.move(*cursor_pos))
        self._last_lines_by_row = current_lines_by_row
        if not self.hide_cursor:
            self.write(self.t.normal_cursor)
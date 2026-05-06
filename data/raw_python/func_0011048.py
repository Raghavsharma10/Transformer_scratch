def get_cursor_vertical_diff(self):
        """Returns the how far down the cursor moved since last render.

        Note:
            If another get_cursor_vertical_diff call is already in progress,
            immediately returns zero. (This situation is likely if
            get_cursor_vertical_diff is called from a SIGWINCH signal
            handler, since sigwinches can happen in rapid succession and
            terminal emulators seem not to respond to cursor position
            queries before the next sigwinch occurs.)
        """
        # Probably called by a SIGWINCH handler, and therefore
        # will do cursor querying until a SIGWINCH doesn't happen during
        # the query. Calls to the function from a signal handler COULD STILL
        # HAPPEN out of order -
        # they just can't interrupt the actual cursor query.
        if self.in_get_cursor_diff:
            self.another_sigwinch = True
            return 0

        cursor_dy = 0
        while True:
            self.in_get_cursor_diff = True
            self.another_sigwinch = False
            cursor_dy += self._get_cursor_vertical_diff_once()
            self.in_get_cursor_diff = False
            if not self.another_sigwinch:
                return cursor_dy
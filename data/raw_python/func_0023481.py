def set_until(self, frame, lineno=None):
        """Stop when the current line number in frame is greater than lineno or
        when returning from frame."""
        if lineno is None:
            lineno = frame.f_lineno + 1
        self._set_stopinfo(frame, lineno)
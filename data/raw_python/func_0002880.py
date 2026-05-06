def _validate_frow(self, frow):
        """Validate frow argument."""
        is_int = isinstance(frow, int) and (not isinstance(frow, bool))
        pexdoc.exh.addai("frow", not (is_int and (frow >= 0)))
        return frow
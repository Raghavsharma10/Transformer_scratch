def _path(self, s):
        """Parse a path."""
        if s.startswith(b'"'):
            if not s.endswith(b'"'):
                self.abort(errors.BadFormat, '?', '?', s)
            else:
                return _unquote_c_string(s[1:-1])
        return s
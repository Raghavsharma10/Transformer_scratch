def _path_pair(self, s):
        """Parse two paths separated by a space."""
        # TODO: handle a space in the first path
        if s.startswith(b'"'):
            parts = s[1:].split(b'" ', 1)
        else:
            parts = s.split(b' ', 1)
        if len(parts) != 2:
            self.abort(errors.BadFormat, '?', '?', s)
        elif parts[1].startswith(b'"') and parts[1].endswith(b'"'):
            parts[1] = parts[1][1:-1]
        elif parts[1].startswith(b'"') or parts[1].endswith(b'"'):
            self.abort(errors.BadFormat, '?', '?', s)
        return [_unquote_c_string(s) for s in parts]
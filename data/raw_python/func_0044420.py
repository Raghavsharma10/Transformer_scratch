def _get_data(self, required_for, section=b'data'):
        """Parse a data section."""
        line = self.next_line()
        if line.startswith(b'data '):
            rest = line[len(b'data '):]
            if rest.startswith(b'<<'):
                return self.read_until(rest[2:])
            else:
                size = int(rest)
                read_bytes = self.read_bytes(size)
                # optional LF after data.
                next_line = self.input.readline()
                self.lineno += 1
                if len(next_line) > 1 or next_line != b'\n':
                    self.push_line(next_line[:-1])
                return read_bytes
        else:
            self.abort(errors.MissingSection, required_for, section)
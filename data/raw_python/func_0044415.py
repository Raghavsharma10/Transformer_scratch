def _get_mark_if_any(self):
        """Parse a mark section."""
        line = self.next_line()
        if line.startswith(b'mark :'):
            return line[len(b'mark :'):]
        else:
            self.push_line(line)
            return None
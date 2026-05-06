def _write_marker(self, indent_string, depth, entry, comment):
        """Write a section marker line"""
        return '%s%s%s%s%s' % (indent_string,
                               self._a_to_u('[' * depth),
                               self._quote(self._decode_element(entry), multiline=False),
                               self._a_to_u(']' * depth),
                               self._decode_element(comment))
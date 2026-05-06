def parse(self, output):
        """
        Find stems for a given text.
        """
        output = self._get_lines_with_stems(output)
        words = self._make_unique(output)
        return self._parse_for_simple_stems(words)
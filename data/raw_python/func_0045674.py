def replace(self, pattern, replacement):
        """
        Replace all instances of a pattern with a replacement.

        Args:
            pattern (str): Pattern to replace
            replacement (str): Text to insert
        """
        for i, line in enumerate(self):
            if pattern in line:
                self[i] = line.replace(pattern, replacement)
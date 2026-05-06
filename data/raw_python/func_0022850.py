def add(self, pattern_txt):
        """Add a pattern to the list.

        Args:
            pattern_txt (str list): the pattern, as a list of lines.
        """
        self.patterns[len(pattern_txt)] = pattern_txt

        low = 0
        high = len(pattern_txt) - 1

        while not pattern_txt[low]:
            low += 1

        while not pattern_txt[high]:
            high -= 1

        min_pattern = pattern_txt[low:high + 1]
        self.min_patterns[len(min_pattern)] = min_pattern
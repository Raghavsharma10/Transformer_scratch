def count_lines(self, code):
        """
        Count lines of code (by default ignores empty lines, but child could override to do more).
        """
        return sum(bool(line.strip()) for line in code.splitlines())
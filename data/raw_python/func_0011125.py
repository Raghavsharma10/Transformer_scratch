def splitlines(self, keepends=False):
        """Return a list of lines, split on newline characters,
        include line boundaries, if keepends is true."""
        lines = self.split('\n')
        return [line+'\n' for line in lines] if keepends else (
               lines if lines[-1] else lines[:-1])
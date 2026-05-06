def readline(self, f):
        """A helper method that only reads uncommented lines"""
        while True:
            line = f.readline()
            if len(line) == 0:
                raise EOFError
            line = line[:line.find('#')]
            line = line.strip()
            if len(line) > 0:
                return line
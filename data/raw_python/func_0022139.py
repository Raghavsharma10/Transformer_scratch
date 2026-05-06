def from_file(self, filename):
        """Update running digest with content of named file."""
        f = open(filename, 'rb')
        while True:
            data = f.read(10480)
            if not data:
                break
            self.update(data)
        f.close()
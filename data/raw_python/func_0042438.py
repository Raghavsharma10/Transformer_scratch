def _data(self):
        """Read data from the file."""
        current_position = self.file.tell()
        self.file.seek(self.position)
        data = self.file.read(self.length)
        self.file.seek(current_position)
        if self.length % 2:
            data += '\x00' # Padding byte
        return data
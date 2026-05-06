def set_properties(self, path, mode):
        """Set file's properties (name and mode).

        This function is also in charge of swapping between textual and
        binary streams.
        """
        self.name = path
        self.mode = mode

        if 'b' in self.mode:
            if not isinstance(self.read_data, bytes):
                self.read_data = bytes(self.read_data, encoding='utf8')
        else:
            if not isinstance(self.read_data, str):
                self.read_data = str(self.read_data, encoding='utf8')
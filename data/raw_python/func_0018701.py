def write(self, out):
        """Used in constructing an outgoing packet"""
        out.write_string(self.text, len(self.text))
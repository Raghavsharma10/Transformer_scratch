def write(self, out):
        """Used in constructing an outgoing packet"""
        out.write_string(self.cpu, len(self.cpu))
        out.write_string(self.os, len(self.os))
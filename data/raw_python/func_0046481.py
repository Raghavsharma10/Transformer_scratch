def port_pair(self):
        """The port and it's transport as a pair"""
        if self.transport is NotSpecified:
            return (self.port, "tcp")
        else:
            return (self.port, self.transport)
def port_str(self):
        """The port and it's transport as a single string"""
        if self.transport is NotSpecified:
            return str(self.port)
        else:
            return "{0}/{1}".format(self.port, self.transport)
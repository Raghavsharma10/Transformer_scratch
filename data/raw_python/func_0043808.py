def connection(self):
        """identify the remote connection parameters"""
        self.getPorts()         # acquire if necessary
        self.getIPaddresses()   # acquire if necessary
        return (self.ipAddress, self.ports)
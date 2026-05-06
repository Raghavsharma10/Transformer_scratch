def getIPaddresses(self):
        """identify the IP addresses where this process client will launch the SC2 client"""
        if not self.ipAddress:
            self.ipAddress = ipAddresses.getAll() # update with IP address
        return self.ipAddress
def disconnect_all(self):
        """
        Disconnects from all connected servers.
        :rtype: self
        """
        addresses = deepcopy(self._addresses)

        for ip, port in addresses:
            self.disconnect(ip, port)

        return self
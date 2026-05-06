def remove_ip(self, IPAddress):
        """
        Release the specified IP-address from the server.
        """
        self.cloud_manager.release_ip(IPAddress.address)
        self.ip_addresses.remove(IPAddress)
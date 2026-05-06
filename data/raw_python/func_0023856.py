def get_ip(self, address):
        """
        Get an IPAddress object with the IP address (string) from the API.

        e.g manager.get_ip('80.69.175.210')
        """
        res = self.get_request('/ip_address/' + address)
        return IPAddress(cloud_manager=self, **res['ip_address'])
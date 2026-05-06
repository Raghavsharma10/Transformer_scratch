def get_ips(self):
        """
        Get all IPAddress objects from the API.
        """
        res = self.get_request('/ip_address')
        IPs = IPAddress._create_ip_address_objs(res['ip_addresses'], cloud_manager=self)
        return IPs
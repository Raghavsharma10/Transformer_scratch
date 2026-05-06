def get_server_by_ip(self, ip_address):
        """
        Return a (populated) Server instance by its IP.

        Uses GET '/ip_address/x.x.x.x' to retrieve machine UUID using IP-address.
        """
        data = self.get_request('/ip_address/{0}'.format(ip_address))
        UUID = data['ip_address']['server']
        return self.get_server(UUID)
def add_ip(self, family='IPv4'):
        """
        Allocate a new (random) IP-address to the Server.
        """
        IP = self.cloud_manager.attach_ip(self.uuid, family)
        self.ip_addresses.append(IP)
        return IP
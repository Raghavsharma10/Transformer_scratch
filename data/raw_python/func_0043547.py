def allow(self, ip):
        """
        Allow the specified IP to connect.

        :param ip: The IPv4 or IPv6 address to allow.
        """
        self.whitelist.append(ip)
        self.blacklist.clear()
def deny(self, ip):
        """
        Allow the specified IP to connect.

        :param ip: The IPv4 or IPv6 address to allow.
        """
        self.blacklist.append(ip)
        self.whitelist.clear()
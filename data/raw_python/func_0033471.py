def add_ipv6addr(self, ipv6addr):
        """Add an IPv6 address to the host.

        :param str ipv6addr: The IP address to add.
        :raises: ValueError

        """
        for addr in self.ipv6addrs:
            if ((isinstance(addr, dict) and addr['ipv6addr'] == ipv6addr) or
                (isinstance(addr, HostIPv4) and addr.ipv6addr == ipv6addr)):
                raise ValueError('Already exists')
        self.ipv6addrs.append({'ipv6addr': ipv6addr})
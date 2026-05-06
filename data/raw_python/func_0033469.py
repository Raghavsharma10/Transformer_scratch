def add_ipv4addr(self, ipv4addr):
        """Add an IPv4 address to the host.

        :param str ipv4addr: The IP address to add.
        :raises: ValueError

        """
        for addr in self.ipv4addrs:
            if ((isinstance(addr, dict) and addr['ipv4addr'] == ipv4addr) or
                (isinstance(addr, HostIPv4) and addr.ipv4addr == ipv4addr)):
                raise ValueError('Already exists')
        self.ipv4addrs.append({'ipv4addr': ipv4addr})
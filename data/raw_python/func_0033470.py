def remove_ipv4addr(self, ipv4addr):
        """Remove an IPv4 address from the host.

        :param str ipv4addr: The IP address to remove

        """
        for addr in self.ipv4addrs:
            if ((isinstance(addr, dict) and addr['ipv4addr'] == ipv4addr) or
                (isinstance(addr, HostIPv4) and addr.ipv4addr == ipv4addr)):
                self.ipv4addrs.remove(addr)
                break
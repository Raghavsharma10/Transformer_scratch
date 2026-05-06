def remove_ipv6addr(self, ipv6addr):
        """Remove an IPv6 address from the host.

        :param str ipv6addr: The IP address to remove

        """
        for addr in self.ipv6addrs:
            if ((isinstance(addr, dict) and addr['ipv6addr'] == ipv6addr) or
                (isinstance(addr, HostIPv4) and addr.ipv6addr == ipv6addr)):
                self.ipv6addrs.remove(addr)
                break
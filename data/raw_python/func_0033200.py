def add_new_host(self, hostname, ipv4addr, comment=None):
        """Add or update a host in the infoblox, overwriting any IP address
        entries.

        :param str hostname: Hostname to add/set
        :param str ipv4addr: IP Address to add/set
        :param str comment: The comment for the record

        """
        host = Host(self.session, name=hostname)
        if host.ipv4addrs:
            host.ipv4addrs = []
        host.add_ipv4addr(ipv4addr)
        host.comment = comment
        return host.save()
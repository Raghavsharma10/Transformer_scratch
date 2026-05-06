def get_ip(self, access='public', addr_family=None, strict=None):
        """
        Return the server's IP address.

        Params:
        - addr_family: IPv4, IPv6 or None. None prefers IPv4 but will
                       return IPv6 if IPv4 addr was not available.
        - access: 'public' or 'private'
        """
        if addr_family not in ['IPv4', 'IPv6', None]:
            raise Exception("`addr_family` must be 'IPv4', 'IPv6' or None")

        if access not in ['private', 'public']:
            raise Exception("`access` must be 'public' or 'private'")

        if not hasattr(self, 'ip_addresses'):
            self.populate()

        # server can have several public or private IPs
        ip_addrs = [
            ip_addr for ip_addr in self.ip_addresses
            if ip_addr.access == access
        ]

        # prefer addr_family (or IPv4 if none given)
        preferred_family = addr_family if addr_family else 'IPv4'
        for ip_addr in ip_addrs:
            if ip_addr.family == preferred_family:
                return ip_addr.address

        # any IP (of the right access) will do if available and addr_family is None
        return ip_addrs[0].address if ip_addrs and not addr_family else None
def parse_ip_address(self, ip_address):
        """Parse an address as returned by the ``connected_users_info`` or ``user_sessions_info`` API calls.

        Example::

            >>> EjabberdBackendBase().parse_ip_address('192.168.0.1')  # doctest: +FORCE_TEXT
            IPv4Address('192.168.0.1')
            >>> EjabberdBackendBase().parse_ip_address('::FFFF:192.168.0.1')  # doctest: +FORCE_TEXT
            IPv4Address('192.168.0.1')
            >>> EjabberdBackendBase().parse_ip_address('::1')  # doctest: +FORCE_TEXT
            IPv6Address('::1')

        :param ip_address: An IP address.
        :type  ip_address: str
        :return: The parsed IP address.
        :rtype: `ipaddress.IPv6Address` or `ipaddress.IPv4Address`.
        """
        if ip_address.startswith('::FFFF:'):
            ip_address = ip_address[7:]
        if six.PY2 and isinstance(ip_address, str):
            # ipaddress constructor does not eat str in py2 :-/
            ip_address = ip_address.decode('utf-8')

        return ipaddress.ip_address(ip_address)
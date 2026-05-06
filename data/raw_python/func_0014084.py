def _get_address_family(table):
        """
        Function to derive address family from a junos table name.

        :params table: The name of the routing table
        :returns: address family
        """
        address_family_mapping = {
            'inet': 'ipv4',
            'inet6': 'ipv6',
            'inetflow': 'flow'
        }
        family = table.split('.')[-2]
        try:
            address_family = address_family_mapping[family]
        except KeyError:
            address_family = family
        return address_family
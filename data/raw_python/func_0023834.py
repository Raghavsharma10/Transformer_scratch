def _create_ip_address_objs(ip_addresses, cloud_manager):
        """
        Create IPAddress objects from API response data.
        Also associates CloudManager with the objects.
        """
        # ip-addresses might be provided as a flat array or as a following dict:
        # {'ip_addresses': {'ip_address': [...]}} || {'ip_address': [...]}

        if 'ip_addresses' in ip_addresses:
            ip_addresses = ip_addresses['ip_addresses']

        if 'ip_address' in ip_addresses:
            ip_addresses = ip_addresses['ip_address']

        return [
            IPAddress(cloud_manager=cloud_manager, **ip_addr)
            for ip_addr in ip_addresses
        ]
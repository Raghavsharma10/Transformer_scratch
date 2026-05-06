def qs_for_ip(cls, ip_str):
        """
        Returns a queryset with matching IPNetwork objects for the given IP.
        """
        ip = int(netaddr.IPAddress(ip_str))

        # ignore IPv6 addresses for now (4294967295 is 0xffffffff, aka the
        # biggest 32-bit number)
        if ip > 4294967295:
            return cls.objects.none()

        ip_range_query = {
            'start__lte': ip,
            'stop__gte': ip
        }

        return cls.objects.filter(**ip_range_query)
def create_from_string(cls, cidr, label=None, whitelist=False):
        """
        Converts a CIDR like 192.168.0.0/24 into 2 parts:
            start: 3232235520
            stop: 3232235775
        """
        network = netaddr.IPNetwork(cidr)
        start = network.first
        stop = start + network.size - 1
        obj = cls.objects.create(label=label, start=start, stop=stop,
                                 whitelist=whitelist)
        return obj
def _get_hosts_from_ports(self, ports):
        """ validate hostnames from a list of ports
        """
        hosts = map(lambda x: 'localhost:%d' % int(x.strip()), ports.split(','))
        return list(set(hosts))
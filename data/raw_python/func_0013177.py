def get_interfaces_ip(self):
        """
        Get interface ip details.

        Returns a dict of dicts

        Example Output:

        {   u'FastEthernet8': {   'ipv4': {   u'10.66.43.169': {   'prefix_length': 22}}},
            u'Loopback555': {   'ipv4': {   u'192.168.1.1': {   'prefix_length': 24}},
                                'ipv6': {   u'1::1': {   'prefix_length': 64},
                                            u'2001:DB8:1::1': {   'prefix_length': 64},
                                            u'2::': {   'prefix_length': 64},
                                            u'FE80::3': {   'prefix_length': 10}}},
            u'Tunnel0': {   'ipv4': {   u'10.63.100.9': {   'prefix_length': 24}}},
            u'Tunnel1': {   'ipv4': {   u'10.63.101.9': {   'prefix_length': 24}}},
            u'Vlan100': {   'ipv4': {   u'10.40.0.1': {   'prefix_length': 24},
                                        u'10.41.0.1': {   'prefix_length': 24},
                                        u'10.65.0.1': {   'prefix_length': 24}}},
            u'Vlan200': {   'ipv4': {   u'10.63.176.57': {   'prefix_length': 29}}}}
        """
        interfaces = {}

        command = 'show ip interface'
        show_ip_interface = self._send_command(command)
        command = 'show ipv6 interface'
        show_ipv6_interface = self._send_command(command)

        INTERNET_ADDRESS = r'\s+(?:Internet address is|Secondary address)'
        INTERNET_ADDRESS += r' (?P<ip>{})/(?P<prefix>\d+)'.format(IPV4_ADDR_REGEX)
        LINK_LOCAL_ADDRESS = r'\s+IPv6 is enabled, link-local address is (?P<ip>[a-fA-F0-9:]+)'
        GLOBAL_ADDRESS = r'\s+(?P<ip>[a-fA-F0-9:]+), subnet is (?:[a-fA-F0-9:]+)/(?P<prefix>\d+)'

        interfaces = {}
        for line in show_ip_interface.splitlines():
            if(len(line.strip()) == 0):
                continue
            if(line[0] != ' '):
                ipv4 = {}
                interface_name = line.split()[0]
            m = re.match(INTERNET_ADDRESS, line)
            if m:
                ip, prefix = m.groups()
                ipv4.update({ip: {"prefix_length": int(prefix)}})
                interfaces[interface_name] = {'ipv4': ipv4}

        for line in show_ipv6_interface.splitlines():
            if(len(line.strip()) == 0):
                continue
            if(line[0] != ' '):
                ifname = line.split()[0]
                ipv6 = {}
                if ifname not in interfaces:
                    interfaces[ifname] = {'ipv6': ipv6}
                else:
                    interfaces[ifname].update({'ipv6': ipv6})
            m = re.match(LINK_LOCAL_ADDRESS, line)
            if m:
                ip = m.group(1)
                ipv6.update({ip: {"prefix_length": 10}})
            m = re.match(GLOBAL_ADDRESS, line)
            if m:
                ip, prefix = m.groups()
                ipv6.update({ip: {"prefix_length": int(prefix)}})

        # Interface without ipv6 doesn't appears in show ipv6 interface
        return interfaces
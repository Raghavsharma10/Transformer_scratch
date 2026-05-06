def get_interfaces(self):
        """
        Get interface details.

        last_flapped is not implemented

        Example Output:

        {   u'Vlan1': {   'description': u'N/A',
                      'is_enabled': True,
                      'is_up': True,
                      'last_flapped': -1.0,
                      'mac_address': u'a493.4cc1.67a7',
                      'speed': 100},
        u'Vlan100': {   'description': u'Data Network',
                        'is_enabled': True,
                        'is_up': True,
                        'last_flapped': -1.0,
                        'mac_address': u'a493.4cc1.67a7',
                        'speed': 100},
        u'Vlan200': {   'description': u'Voice Network',
                        'is_enabled': True,
                        'is_up': True,
                        'last_flapped': -1.0,
                        'mac_address': u'a493.4cc1.67a7',
                        'speed': 100}}
        """
        # default values.
        last_flapped = -1.0

        command = 'show interfaces'
        output = self._send_command(command)

        interface = description = mac_address = speed = speedformat = ''
        is_enabled = is_up = None

        interface_dict = {}
        for line in output.splitlines():

            interface_regex_1 = r"^(\S+?)\s+is\s+(.+?),\s+line\s+protocol\s+is\s+(\S+)"
            interface_regex_2 = r"^(\S+)\s+is\s+(up|down)"
            for pattern in (interface_regex_1, interface_regex_2):
                interface_match = re.search(pattern, line)
                if interface_match:
                    interface = interface_match.group(1)
                    status = interface_match.group(2)
                    try:
                        protocol = interface_match.group(3)
                    except IndexError:
                        protocol = ''
                    if 'admin' in status.lower():
                        is_enabled = False
                    else:
                        is_enabled = True
                    if protocol:
                        is_up = bool('up' in protocol)
                    else:
                        is_up = bool('up' in status)
                    break

            mac_addr_regex = r"^\s+Hardware.+address\s+is\s+({})".format(MAC_REGEX)
            if re.search(mac_addr_regex, line):
                mac_addr_match = re.search(mac_addr_regex, line)
                mac_address = napalm_base.helpers.mac(mac_addr_match.groups()[0])

            descr_regex = "^\s+Description:\s+(.+?)$"
            if re.search(descr_regex, line):
                descr_match = re.search(descr_regex, line)
                description = descr_match.groups()[0]

            speed_regex = r"^\s+MTU\s+\d+.+BW\s+(\d+)\s+([KMG]?b)"
            if re.search(speed_regex, line):
                speed_match = re.search(speed_regex, line)
                speed = speed_match.groups()[0]
                speedformat = speed_match.groups()[1]
                speed = float(speed)
                if speedformat.startswith('Kb'):
                    speed = speed / 1000.0
                elif speedformat.startswith('Gb'):
                    speed = speed * 1000
                speed = int(round(speed))

                if interface == '':
                    raise ValueError("Interface attributes were \
                                      found without any known interface")
                if not isinstance(is_up, bool) or not isinstance(is_enabled, bool):
                    raise ValueError("Did not correctly find the interface status")

                interface_dict[interface] = {'is_enabled': is_enabled, 'is_up': is_up,
                                             'description': description, 'mac_address': mac_address,
                                             'last_flapped': last_flapped, 'speed': speed}

                interface = description = mac_address = speed = speedformat = ''
                is_enabled = is_up = None

        return interface_dict
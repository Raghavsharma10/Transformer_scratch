def get_arp_table(self):
        """
        Get arp table information.

        Return a list of dictionaries having the following set of keys:
            * interface (string)
            * mac (string)
            * ip (string)
            * age (float)

        For example::
            [
                {
                    'interface' : 'MgmtEth0/RSP0/CPU0/0',
                    'mac'       : '5c:5e:ab:da:3c:f0',
                    'ip'        : '172.17.17.1',
                    'age'       : 1454496274.84
                },
                {
                    'interface': 'MgmtEth0/RSP0/CPU0/0',
                    'mac'       : '66:0e:94:96:e0:ff',
                    'ip'        : '172.17.17.2',
                    'age'       : 1435641582.49
                }
            ]
        """
        arp_table = []

        command = 'show arp | exclude Incomplete'
        output = self._send_command(command)

        # Skip the first line which is a header
        output = output.split('\n')
        output = output[1:]

        for line in output:
            if len(line) == 0:
                return {}
            if len(line.split()) == 5:
                # Static ARP entries have no interface
                # Internet  10.0.0.1                -   0010.2345.1cda  ARPA
                interface = ''
                protocol, address, age, mac, eth_type = line.split()
            elif len(line.split()) == 6:
                protocol, address, age, mac, eth_type, interface = line.split()
            else:
                raise ValueError("Unexpected output from: {}".format(line.split()))

            try:
                if age == '-':
                    age = 0
                age = float(age)
            except ValueError:
                raise ValueError("Unable to convert age value to float: {}".format(age))

            # Validate we matched correctly
            if not re.search(RE_IPADDR, address):
                raise ValueError("Invalid IP Address detected: {}".format(address))
            if not re.search(RE_MAC, mac):
                raise ValueError("Invalid MAC Address detected: {}".format(mac))
            entry = {
                'interface': interface,
                'mac': napalm_base.helpers.mac(mac),
                'ip': address,
                'age': age
            }
            arp_table.append(entry)
        return arp_table
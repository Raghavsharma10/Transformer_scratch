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
                    'age'       : 12.0
                },
                {
                    'interface': 'MgmtEth0/RSP0/CPU0/0',
                    'mac'       : '66:0e:94:96:e0:ff',
                    'ip'        : '172.17.17.2',
                    'age'       : 14.0
                }
            ]
        """
        arp_table = []

        command = 'show ip arp vrf default | exc INCOMPLETE'
        output = self.device.send_command(command)

        separator = r"^Address\s+Age.*Interface.*$"
        arp_list = re.split(separator, output, flags=re.M)
        if len(arp_list) != 2:
            raise ValueError("Error processing arp table output:\n\n{}".format(output))

        arp_entries = arp_list[1].strip()
        for line in arp_entries.splitlines():
            if len(line.split()) == 4:
                address, age, mac, interface = line.split()
            else:
                raise ValueError("Unexpected output from: {}".format(line.split()))

            if age == '-':
                age = -1.0
            elif ':' not in age:
                # Cisco sometimes returns a sub second arp time 0.411797
                try:
                    age = float(age)
                except ValueError:
                    age = -1.0
            else:
                age = convert_hhmmss(age)
                age = float(age)
            age = round(age, 1)

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
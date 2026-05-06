def get_mac_address_table(self):
        """
        Returns a lists of dictionaries. Each dictionary represents an entry in the MAC Address
        Table, having the following keys
            * mac (string)
            * interface (string)
            * vlan (int)
            * active (boolean)
            * static (boolean)
            * moves (int)
            * last_move (float)
        Format1:

        Legend:
        * - primary entry, G - Gateway MAC, (R) - Routed MAC, O - Overlay MAC
        age - seconds since last seen,+ - primary entry using vPC Peer-Link,
        (T) - True, (F) - False
           VLAN     MAC Address      Type      age     Secure NTFY Ports/SWID.SSID.LID
        ---------+-----------------+--------+---------+------+----+------------------
        * 27       0026.f064.0000    dynamic      -       F    F    po1
        * 27       001b.54c2.2644    dynamic      -       F    F    po1
        * 27       0000.0c9f.f2bc    dynamic      -       F    F    po1
        * 27       0026.980a.df44    dynamic      -       F    F    po1
        * 16       0050.56bb.0164    dynamic      -       F    F    po2
        * 13       90e2.ba5a.9f30    dynamic      -       F    F    eth1/2
        * 13       90e2.ba4b.fc78    dynamic      -       F    F    eth1/1
        """

        #  The '*' is stripped out later
        RE_MACTABLE_FORMAT1 = r"^\s+{}\s+{}\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+".format(VLAN_REGEX,
                                                                                  MAC_REGEX)
        RE_MACTABLE_FORMAT2 = r"^\s+{}\s+{}\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+".format('-',
                                                                                  MAC_REGEX)

        mac_address_table = []
        command = 'show mac address-table'
        output = self.device.send_command(command) # noqa

        def remove_prefix(s, prefix):
            return s[len(prefix):] if s.startswith(prefix) else s

        def process_mac_fields(vlan, mac, mac_type, interface):
            """Return proper data for mac address fields."""
            if mac_type.lower() in ['self', 'static', 'system']:
                static = True
                if vlan.lower() == 'all':
                    vlan = 0
                elif vlan == '-':
                    vlan = 0
                if interface.lower() == 'cpu' or re.search(r'router', interface.lower()) or \
                        re.search(r'switch', interface.lower()):
                    interface = ''
            else:
                static = False
            if mac_type.lower() in ['dynamic']:
                active = True
            else:
                active = False
            return {
                'mac': napalm_base.helpers.mac(mac),
                'interface': interface,
                'vlan': int(vlan),
                'static': static,
                'active': active,
                'moves': -1,
                'last_move': -1.0
            }

        # Skip the header lines
        output = re.split(r'^----.*', output, flags=re.M)[1:]
        output = "\n".join(output).strip()
        # Strip any leading astericks or G character
        output = re.sub(r"^[\*G]", "", output, flags=re.M)

        for line in output.splitlines():

            # Every 500 Mac's Legend is reprinted, regardless of term len.
            # Above split will not help in this scenario
            if re.search('^Legend', line):
                continue
            elif re.search('^\s+\* \- primary entry', line):
                continue
            elif re.search('^\s+age \-', line):
                continue
            elif re.search('^\s+VLAN', line):
                continue
            elif re.search('^------', line):
                continue
            elif re.search('^\s*$', line):
                continue

            for pattern in [RE_MACTABLE_FORMAT1, RE_MACTABLE_FORMAT2]:
                if re.search(pattern, line):
                    if len(line.split()) == 7:
                        vlan, mac, mac_type, _, _, _, interface = line.split()
                        mac_address_table.append(process_mac_fields(vlan, mac, mac_type, interface))
                        break
            else:
                raise ValueError("Unexpected output from: {}".format(repr(line)))

        return mac_address_table
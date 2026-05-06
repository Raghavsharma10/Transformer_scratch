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
        Destination Address  Address Type  VLAN  Destination Port
        -------------------  ------------  ----  --------------------
        6400.f1cf.2cc6          Dynamic       1     Wlan-GigabitEthernet0

        Cat 6500:
        Legend: * - primary entry
                age - seconds since last seen
                n/a - not available

          vlan   mac address     type    learn     age              ports
        ------+----------------+--------+-----+----------+--------------------------
        *  999  1111.2222.3333   dynamic  Yes          0   Port-channel1
           999  1111.2222.3333   dynamic  Yes          0   Port-channel1

        Cat 4948
        Unicast Entries
         vlan   mac address     type        protocols               port
        -------+---------------+--------+---------------------+--------------------
         999    1111.2222.3333   dynamic ip                    Port-channel1

        Cat 2960
        Mac Address Table
        -------------------------------------------

        Vlan    Mac Address       Type        Ports
        ----    -----------       --------    -----
        All    1111.2222.3333    STATIC      CPU
        """

        RE_MACTABLE_DEFAULT = r"^" + MAC_REGEX
        RE_MACTABLE_6500_1 = r"^\*\s+{}\s+{}\s+".format(VLAN_REGEX, MAC_REGEX)  # 7 fields
        RE_MACTABLE_6500_2 = r"^{}\s+{}\s+".format(VLAN_REGEX, MAC_REGEX)   # 6 fields
        RE_MACTABLE_6500_3 = r"^\s{51}\S+"                               # Fill down from prior
        RE_MACTABLE_4500_1 = r"^{}\s+{}\s+".format(VLAN_REGEX, MAC_REGEX)     # 5 fields
        RE_MACTABLE_4500_2 = r"^\s{32}\S+"                               # Fill down from prior
        RE_MACTABLE_2960_1 = r"^All\s+{}".format(MAC_REGEX)
        RE_MACTABLE_GEN_1 = r"^{}\s+{}\s+".format(VLAN_REGEX, MAC_REGEX)   # 4 fields (2960/4500)

        def process_mac_fields(vlan, mac, mac_type, interface):
            """Return proper data for mac address fields."""
            if mac_type.lower() in ['self', 'static', 'system']:
                static = True
                if vlan.lower() == 'all':
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

        mac_address_table = []
        command = IOS_COMMANDS['show_mac_address']
        output = self._send_command(command)

        # Skip the header lines
        output = re.split(r'^----.*', output, flags=re.M)[1:]
        output = "\n".join(output).strip()
        # Strip any leading astericks
        output = re.sub(r"^\*", "", output, flags=re.M)
        fill_down_vlan = fill_down_mac = fill_down_mac_type = ''
        for line in output.splitlines():
            # Cat6500 one off anf 4500 multicast format
            if (re.search(RE_MACTABLE_6500_3, line) or re.search(RE_MACTABLE_4500_2, line)):
                interface = line.strip()
                if ',' in interface:
                    interfaces = interface.split(',')
                else:
                    interfaces = []
                    interfaces.append(interface)
                for single_interface in interfaces:
                    mac_address_table.append(process_mac_fields(fill_down_vlan, fill_down_mac,
                                                                fill_down_mac_type,
                                                                single_interface))
                continue
            line = line.strip()
            if line == '':
                continue
            if re.search(r"^---", line):
                # Convert any '---' to VLAN 0
                line = re.sub(r"^---", "0", line, flags=re.M)

            # Format1
            if re.search(RE_MACTABLE_DEFAULT, line):
                if len(line.split()) == 4:
                    mac, mac_type, vlan, interface = line.split()
                    mac_address_table.append(process_mac_fields(vlan, mac, mac_type, interface))
                else:
                    raise ValueError("Unexpected output from: {}".format(line.split()))
            # Cat6500 format
            elif (re.search(RE_MACTABLE_6500_1, line) or re.search(RE_MACTABLE_6500_2, line)) and \
                    len(line.split()) >= 6:
                if len(line.split()) == 7:
                    _, vlan, mac, mac_type, _, _, interface = line.split()
                elif len(line.split()) == 6:
                    vlan, mac, mac_type, _, _, interface = line.split()
                if ',' in interface:
                    interfaces = interface.split(',')
                    fill_down_vlan = vlan
                    fill_down_mac = mac
                    fill_down_mac_type = mac_type
                    for single_interface in interfaces:
                        mac_address_table.append(process_mac_fields(vlan, mac, mac_type,
                                                                    single_interface))
                else:
                    mac_address_table.append(process_mac_fields(vlan, mac, mac_type, interface))
            # Cat4500 format
            elif re.search(RE_MACTABLE_4500_1, line) and len(line.split()) == 5:
                vlan, mac, mac_type, _, interface = line.split()
                mac_address_table.append(process_mac_fields(vlan, mac, mac_type, interface))
            # Cat2960 format - ignore extra header line
            elif re.search(r"^Vlan\s+Mac Address\s+", line):
                continue
            # Cat2960 format (Cat4500 format multicast entries)
            elif (re.search(RE_MACTABLE_2960_1, line) or re.search(RE_MACTABLE_GEN_1, line)) and \
                    len(line.split()) == 4:
                vlan, mac, mac_type, interface = line.split()
                if ',' in interface:
                    interfaces = interface.split(',')
                    fill_down_vlan = vlan
                    fill_down_mac = mac
                    fill_down_mac_type = mac_type
                    for single_interface in interfaces:
                        mac_address_table.append(process_mac_fields(vlan, mac, mac_type,
                                                                    single_interface))
                else:
                    mac_address_table.append(process_mac_fields(vlan, mac, mac_type, interface))
            elif re.search(r"Total Mac Addresses", line):
                continue
            elif re.search(r"Multicast Entries", line):
                continue
            elif re.search(r"vlan.*mac.*address.*type.*", line):
                continue
            else:
                raise ValueError("Unexpected output from: {}".format(repr(line)))

        return mac_address_table
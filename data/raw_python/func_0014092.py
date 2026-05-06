def get_mac_address_table(self):
        """Return the MAC address table."""
        mac_address_table = []

        if self.device.facts.get('personality', '') in ['SWITCH']:  # for EX & QFX devices
            if self.device.facts.get('switch_style', '') in ['VLAN_L2NG']:  # for L2NG devices
                mac_table = junos_views.junos_mac_address_table_switch_l2ng(self.device)
            else:
                mac_table = junos_views.junos_mac_address_table_switch(self.device)
        else:
            mac_table = junos_views.junos_mac_address_table(self.device)

        mac_table.get()
        mac_table_items = mac_table.items()

        default_values = {
            'mac': u'',
            'interface': u'',
            'vlan': 0,
            'static': False,
            'active': True,
            'moves': 0,
            'last_move': 0.0
        }

        for mac_table_entry in mac_table_items:
            mac_entry = default_values.copy()
            mac_entry.update(
                {elem[0]: elem[1] for elem in mac_table_entry[1]}
            )
            mac = mac_entry.get('mac')

            # JUNOS returns '*' for Type = Flood
            if mac == '*':
                continue

            mac_entry['mac'] = napalm_base.helpers.mac(mac)
            mac_address_table.append(mac_entry)

        return mac_address_table
def get_arp_table(self):
        """Return the ARP table."""
        # could use ArpTable
        # from jnpr.junos.op.phyport import ArpTable
        # and simply use it
        # but
        # we need:
        #   - filters
        #   - group by VLAN ID
        #   - hostname & TTE fields as well

        arp_table = []

        arp_table_raw = junos_views.junos_arp_table(self.device)
        arp_table_raw.get()
        arp_table_items = arp_table_raw.items()

        for arp_table_entry in arp_table_items:
            arp_entry = {
                elem[0]: elem[1] for elem in arp_table_entry[1]
            }
            arp_entry['mac'] = napalm_base.helpers.mac(arp_entry.get('mac'))
            arp_entry['ip'] = napalm_base.helpers.ip(arp_entry.get('ip'))
            arp_table.append(arp_entry)

        return arp_table
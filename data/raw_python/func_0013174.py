def get_lldp_neighbors_detail(self, interface=''):
        """
        IOS implementation of get_lldp_neighbors_detail.

        Calls get_lldp_neighbors.
        """
        lldp = {}
        lldp_neighbors = self.get_lldp_neighbors()

        # Filter to specific interface
        if interface:
            lldp_data = lldp_neighbors.get(interface)
            if lldp_data:
                lldp_neighbors = {interface: lldp_data}
            else:
                lldp_neighbors = {}

        for interface in lldp_neighbors:
            local_port = interface
            lldp_fields = self._lldp_detail_parser(interface)
            # Convert any 'not advertised' to 'N/A'
            for field in lldp_fields:
                for i, value in enumerate(field):
                    if 'not advertised' in value:
                        field[i] = 'N/A'
            number_entries = len(lldp_fields[0])

            # re.findall will return a list. Make sure same number of entries always returned.
            for test_list in lldp_fields:
                if len(test_list) != number_entries:
                    raise ValueError("Failure processing show lldp neighbors detail")

            # Standardize the fields
            port_id, port_description, chassis_id, system_name, system_description, \
                system_capabilities, enabled_capabilities, remote_address = lldp_fields
            standardized_fields = zip(port_id, port_description, chassis_id, system_name,
                                      system_description, system_capabilities,
                                      enabled_capabilities, remote_address)

            lldp.setdefault(local_port, [])
            for entry in standardized_fields:
                remote_port_id, remote_port_description, remote_chassis_id, remote_system_name, \
                    remote_system_description, remote_system_capab, remote_enabled_capab, \
                    remote_mgmt_address = entry

                lldp[local_port].append({
                    'parent_interface': u'N/A',
                    'remote_port': remote_port_id,
                    'remote_port_description': remote_port_description,
                    'remote_chassis_id': remote_chassis_id,
                    'remote_system_name': remote_system_name,
                    'remote_system_description': remote_system_description,
                    'remote_system_capab': remote_system_capab,
                    'remote_system_enable_capab': remote_enabled_capab})

        return lldp
def mac_group_mac_group_entry_entry_address(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_group = ET.SubElement(config, "mac-group", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        mac_group_id_key = ET.SubElement(mac_group, "mac-group-id")
        mac_group_id_key.text = kwargs.pop('mac_group_id')
        mac_group_entry = ET.SubElement(mac_group, "mac-group-entry")
        entry_address = ET.SubElement(mac_group_entry, "entry-address")
        entry_address.text = kwargs.pop('entry_address')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
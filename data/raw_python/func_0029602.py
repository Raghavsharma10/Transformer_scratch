def mac_group_mac_group_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_group = ET.SubElement(config, "mac-group", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        mac_group_id = ET.SubElement(mac_group, "mac-group-id")
        mac_group_id.text = kwargs.pop('mac_group_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
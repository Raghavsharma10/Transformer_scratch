def mac_address_table_mac_move_mac_move_detect_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        mac_move = ET.SubElement(mac_address_table, "mac-move")
        mac_move_detect_enable = ET.SubElement(mac_move, "mac-move-detect-enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
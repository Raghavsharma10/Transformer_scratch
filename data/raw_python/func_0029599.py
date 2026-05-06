def mac_address_table_mac_move_mac_move_limit(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        mac_move = ET.SubElement(mac_address_table, "mac-move")
        mac_move_limit = ET.SubElement(mac_move, "mac-move-limit")
        mac_move_limit.text = kwargs.pop('mac_move_limit')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
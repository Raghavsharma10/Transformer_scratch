def mac_address_table_learning_mode(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        learning_mode = ET.SubElement(mac_address_table, "learning-mode")
        learning_mode.text = kwargs.pop('learning_mode')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
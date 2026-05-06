def mac_address_table_aging_time_legacy_time_out(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        aging_time = ET.SubElement(mac_address_table, "aging-time")
        legacy_time_out = ET.SubElement(aging_time, "legacy-time-out")
        legacy_time_out.text = kwargs.pop('legacy_time_out')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
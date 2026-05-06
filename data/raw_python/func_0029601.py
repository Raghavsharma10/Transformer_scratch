def mac_address_table_consistency_check_mac_consistency_check_interval(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac_address_table = ET.SubElement(config, "mac-address-table", xmlns="urn:brocade.com:mgmt:brocade-mac-address-table")
        consistency_check = ET.SubElement(mac_address_table, "consistency-check")
        mac_consistency_check_interval = ET.SubElement(consistency_check, "mac-consistency-check-interval")
        mac_consistency_check_interval.text = kwargs.pop('mac_consistency_check_interval')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
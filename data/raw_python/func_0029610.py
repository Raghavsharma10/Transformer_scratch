def get_mac_address_table_output_mac_address_table_mac_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        output = ET.SubElement(get_mac_address_table, "output")
        mac_address_table = ET.SubElement(output, "mac-address-table")
        vlanid_key = ET.SubElement(mac_address_table, "vlanid")
        vlanid_key.text = kwargs.pop('vlanid')
        mac_address_key = ET.SubElement(mac_address_table, "mac-address")
        mac_address_key.text = kwargs.pop('mac_address')
        mac_type = ET.SubElement(mac_address_table, "mac-type")
        mac_type.text = kwargs.pop('mac_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
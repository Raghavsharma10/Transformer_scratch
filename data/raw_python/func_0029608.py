def get_mac_address_table_output_mac_address_table_vlanid(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        output = ET.SubElement(get_mac_address_table, "output")
        mac_address_table = ET.SubElement(output, "mac-address-table")
        mac_address_key = ET.SubElement(mac_address_table, "mac-address")
        mac_address_key.text = kwargs.pop('mac_address')
        vlanid = ET.SubElement(mac_address_table, "vlanid")
        vlanid.text = kwargs.pop('vlanid')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
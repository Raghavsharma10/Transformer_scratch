def get_mac_address_table_output_has_more(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        output = ET.SubElement(get_mac_address_table, "output")
        has_more = ET.SubElement(output, "has-more")
        has_more.text = kwargs.pop('has_more')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
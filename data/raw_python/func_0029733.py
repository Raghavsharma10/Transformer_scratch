def get_mac_address_table_input_request_type_get_next_request_mac_address_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        input = ET.SubElement(get_mac_address_table, "input")
        request_type = ET.SubElement(input, "request-type")
        get_next_request = ET.SubElement(request_type, "get-next-request")
        mac_address_type = ET.SubElement(get_next_request, "mac-address-type")
        mac_address_type.text = kwargs.pop('mac_address_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
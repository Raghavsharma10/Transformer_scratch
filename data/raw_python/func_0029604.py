def get_mac_address_table_input_request_type_get_request_mac_address(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        input = ET.SubElement(get_mac_address_table, "input")
        request_type = ET.SubElement(input, "request-type")
        get_request = ET.SubElement(request_type, "get-request")
        mac_address = ET.SubElement(get_request, "mac-address")
        mac_address.text = kwargs.pop('mac_address')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
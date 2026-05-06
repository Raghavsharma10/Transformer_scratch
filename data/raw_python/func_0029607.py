def get_mac_address_table_input_request_type_get_next_request_last_mac_address_details_last_mac_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        input = ET.SubElement(get_mac_address_table, "input")
        request_type = ET.SubElement(input, "request-type")
        get_next_request = ET.SubElement(request_type, "get-next-request")
        last_mac_address_details = ET.SubElement(get_next_request, "last-mac-address-details")
        last_mac_type = ET.SubElement(last_mac_address_details, "last-mac-type")
        last_mac_type.text = kwargs.pop('last_mac_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
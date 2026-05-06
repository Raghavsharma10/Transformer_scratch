def get_mac_address_table_input_request_type_get_next_request_forwarding_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        input = ET.SubElement(get_mac_address_table, "input")
        request_type = ET.SubElement(input, "request-type")
        get_next_request = ET.SubElement(request_type, "get-next-request")
        forwarding_interface_type = ET.SubElement(get_next_request, "forwarding-interface-type")
        forwarding_interface_type.text = kwargs.pop('forwarding_interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
def get_mac_address_table_input_request_type_get_interface_based_request_forwarding_interface_interface_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_address_table = ET.Element("get_mac_address_table")
        config = get_mac_address_table
        input = ET.SubElement(get_mac_address_table, "input")
        request_type = ET.SubElement(input, "request-type")
        get_interface_based_request = ET.SubElement(request_type, "get-interface-based-request")
        forwarding_interface = ET.SubElement(get_interface_based_request, "forwarding-interface")
        interface_name = ET.SubElement(forwarding_interface, "interface-name")
        interface_name.text = kwargs.pop('interface_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
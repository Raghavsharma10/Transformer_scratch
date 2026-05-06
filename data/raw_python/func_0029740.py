def get_mac_acl_for_intf_input_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_acl_for_intf = ET.Element("get_mac_acl_for_intf")
        config = get_mac_acl_for_intf
        input = ET.SubElement(get_mac_acl_for_intf, "input")
        interface_type = ET.SubElement(input, "interface-type")
        interface_type.text = kwargs.pop('interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
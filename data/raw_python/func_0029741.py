def get_mac_acl_for_intf_input_interface_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_acl_for_intf = ET.Element("get_mac_acl_for_intf")
        config = get_mac_acl_for_intf
        input = ET.SubElement(get_mac_acl_for_intf, "input")
        interface_name = ET.SubElement(input, "interface-name")
        interface_name.text = kwargs.pop('interface_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
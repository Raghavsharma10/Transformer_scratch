def get_mac_acl_for_intf_output_interface_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_acl_for_intf = ET.Element("get_mac_acl_for_intf")
        config = get_mac_acl_for_intf
        output = ET.SubElement(get_mac_acl_for_intf, "output")
        interface = ET.SubElement(output, "interface")
        interface_name_key = ET.SubElement(interface, "interface-name")
        interface_name_key.text = kwargs.pop('interface_name')
        interface_type = ET.SubElement(interface, "interface-type")
        interface_type.text = kwargs.pop('interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
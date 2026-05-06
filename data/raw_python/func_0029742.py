def get_mac_acl_for_intf_input_direction(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_mac_acl_for_intf = ET.Element("get_mac_acl_for_intf")
        config = get_mac_acl_for_intf
        input = ET.SubElement(get_mac_acl_for_intf, "input")
        direction = ET.SubElement(input, "direction")
        direction.text = kwargs.pop('direction')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
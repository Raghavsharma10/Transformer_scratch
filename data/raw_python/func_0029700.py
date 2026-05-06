def get_vmpolicy_macaddr_input_mac(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        input = ET.SubElement(get_vmpolicy_macaddr, "input")
        mac = ET.SubElement(input, "mac")
        mac.text = kwargs.pop('mac')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
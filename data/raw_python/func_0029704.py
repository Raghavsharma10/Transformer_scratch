def get_vmpolicy_macaddr_output_vmpolicy_macaddr_mac(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        output = ET.SubElement(get_vmpolicy_macaddr, "output")
        vmpolicy_macaddr = ET.SubElement(output, "vmpolicy-macaddr")
        mac = ET.SubElement(vmpolicy_macaddr, "mac")
        mac.text = kwargs.pop('mac')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
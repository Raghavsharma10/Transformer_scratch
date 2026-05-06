def get_vmpolicy_macaddr_output_vmpolicy_macaddr_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        output = ET.SubElement(get_vmpolicy_macaddr, "output")
        vmpolicy_macaddr = ET.SubElement(output, "vmpolicy-macaddr")
        name = ET.SubElement(vmpolicy_macaddr, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
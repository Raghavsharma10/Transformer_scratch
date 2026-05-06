def get_vmpolicy_macaddr_input_datacenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        input = ET.SubElement(get_vmpolicy_macaddr, "input")
        datacenter = ET.SubElement(input, "datacenter")
        datacenter.text = kwargs.pop('datacenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
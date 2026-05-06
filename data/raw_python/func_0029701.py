def get_vmpolicy_macaddr_input_vcenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        input = ET.SubElement(get_vmpolicy_macaddr, "input")
        vcenter = ET.SubElement(input, "vcenter")
        vcenter.text = kwargs.pop('vcenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
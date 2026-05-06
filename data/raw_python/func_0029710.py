def get_vmpolicy_macaddr_output_has_more(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        output = ET.SubElement(get_vmpolicy_macaddr, "output")
        has_more = ET.SubElement(output, "has-more")
        has_more.text = kwargs.pop('has_more')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
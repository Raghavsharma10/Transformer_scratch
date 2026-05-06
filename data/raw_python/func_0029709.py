def get_vmpolicy_macaddr_output_vmpolicy_macaddr_port_prof(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        output = ET.SubElement(get_vmpolicy_macaddr, "output")
        vmpolicy_macaddr = ET.SubElement(output, "vmpolicy-macaddr")
        port_prof = ET.SubElement(vmpolicy_macaddr, "port-prof")
        port_prof.text = kwargs.pop('port_prof')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
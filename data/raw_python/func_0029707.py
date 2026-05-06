def get_vmpolicy_macaddr_output_vmpolicy_macaddr_dvpg_nn(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vmpolicy_macaddr = ET.Element("get_vmpolicy_macaddr")
        config = get_vmpolicy_macaddr
        output = ET.SubElement(get_vmpolicy_macaddr, "output")
        vmpolicy_macaddr = ET.SubElement(output, "vmpolicy-macaddr")
        dvpg_nn = ET.SubElement(vmpolicy_macaddr, "dvpg-nn")
        dvpg_nn.text = kwargs.pop('dvpg_nn')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
def get_vnetwork_dvpgs_output_vnetwork_dvpgs_vlan(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        output = ET.SubElement(get_vnetwork_dvpgs, "output")
        vnetwork_dvpgs = ET.SubElement(output, "vnetwork-dvpgs")
        vlan = ET.SubElement(vnetwork_dvpgs, "vlan")
        vlan.text = kwargs.pop('vlan')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
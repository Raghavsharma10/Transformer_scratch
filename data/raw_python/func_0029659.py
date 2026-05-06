def get_vnetwork_dvpgs_output_vnetwork_dvpgs_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        output = ET.SubElement(get_vnetwork_dvpgs, "output")
        vnetwork_dvpgs = ET.SubElement(output, "vnetwork-dvpgs")
        name = ET.SubElement(vnetwork_dvpgs, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
def get_vnetwork_dvs_output_vnetwork_dvs_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvs = ET.Element("get_vnetwork_dvs")
        config = get_vnetwork_dvs
        output = ET.SubElement(get_vnetwork_dvs, "output")
        vnetwork_dvs = ET.SubElement(output, "vnetwork-dvs")
        name = ET.SubElement(vnetwork_dvs, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
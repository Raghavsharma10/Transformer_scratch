def get_vnetwork_dvs_output_vnetwork_dvs_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvs = ET.Element("get_vnetwork_dvs")
        config = get_vnetwork_dvs
        output = ET.SubElement(get_vnetwork_dvs, "output")
        vnetwork_dvs = ET.SubElement(output, "vnetwork-dvs")
        interface_type = ET.SubElement(vnetwork_dvs, "interface-type")
        interface_type.text = kwargs.pop('interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
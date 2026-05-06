def get_vnetwork_vswitches_output_vnetwork_vswitches_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vswitches = ET.Element("get_vnetwork_vswitches")
        config = get_vnetwork_vswitches
        output = ET.SubElement(get_vnetwork_vswitches, "output")
        vnetwork_vswitches = ET.SubElement(output, "vnetwork-vswitches")
        interface_type = ET.SubElement(vnetwork_vswitches, "interface-type")
        interface_type.text = kwargs.pop('interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
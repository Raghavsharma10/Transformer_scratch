def get_vnetwork_vswitches_output_vnetwork_vswitches_host(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vswitches = ET.Element("get_vnetwork_vswitches")
        config = get_vnetwork_vswitches
        output = ET.SubElement(get_vnetwork_vswitches, "output")
        vnetwork_vswitches = ET.SubElement(output, "vnetwork-vswitches")
        host = ET.SubElement(vnetwork_vswitches, "host")
        host.text = kwargs.pop('host')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
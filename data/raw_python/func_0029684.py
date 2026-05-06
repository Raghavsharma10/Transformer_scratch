def get_vnetwork_vswitches_output_vnetwork_vswitches_pnic(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vswitches = ET.Element("get_vnetwork_vswitches")
        config = get_vnetwork_vswitches
        output = ET.SubElement(get_vnetwork_vswitches, "output")
        vnetwork_vswitches = ET.SubElement(output, "vnetwork-vswitches")
        pnic = ET.SubElement(vnetwork_vswitches, "pnic")
        pnic.text = kwargs.pop('pnic')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
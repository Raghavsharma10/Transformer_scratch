def get_vnetwork_vms_output_vnetwork_vms_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        vnetwork_vms = ET.SubElement(output, "vnetwork-vms")
        name = ET.SubElement(vnetwork_vms, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
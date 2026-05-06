def get_vnetwork_vms_output_vnetwork_vms_mac(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        vnetwork_vms = ET.SubElement(output, "vnetwork-vms")
        mac = ET.SubElement(vnetwork_vms, "mac")
        mac.text = kwargs.pop('mac')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
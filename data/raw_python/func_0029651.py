def get_vnetwork_vms_output_vnetwork_vms_ip(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        vnetwork_vms = ET.SubElement(output, "vnetwork-vms")
        ip = ET.SubElement(vnetwork_vms, "ip")
        ip.text = kwargs.pop('ip')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
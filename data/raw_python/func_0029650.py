def get_vnetwork_vms_output_vnetwork_vms_datacenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        vnetwork_vms = ET.SubElement(output, "vnetwork-vms")
        datacenter = ET.SubElement(vnetwork_vms, "datacenter")
        datacenter.text = kwargs.pop('datacenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
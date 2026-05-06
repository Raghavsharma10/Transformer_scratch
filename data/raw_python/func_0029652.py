def get_vnetwork_vms_output_vnetwork_vms_host_nn(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        vnetwork_vms = ET.SubElement(output, "vnetwork-vms")
        host_nn = ET.SubElement(vnetwork_vms, "host-nn")
        host_nn.text = kwargs.pop('host_nn')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
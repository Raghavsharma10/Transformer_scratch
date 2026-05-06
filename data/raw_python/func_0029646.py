def get_vnetwork_vms_input_datacenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        input = ET.SubElement(get_vnetwork_vms, "input")
        datacenter = ET.SubElement(input, "datacenter")
        datacenter.text = kwargs.pop('datacenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
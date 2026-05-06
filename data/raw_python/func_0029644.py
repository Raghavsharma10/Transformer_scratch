def get_vnetwork_vms_input_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        input = ET.SubElement(get_vnetwork_vms, "input")
        name = ET.SubElement(input, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
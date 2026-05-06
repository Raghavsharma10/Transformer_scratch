def get_vnetwork_vswitches_input_vcenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vswitches = ET.Element("get_vnetwork_vswitches")
        config = get_vnetwork_vswitches
        input = ET.SubElement(get_vnetwork_vswitches, "input")
        vcenter = ET.SubElement(input, "vcenter")
        vcenter.text = kwargs.pop('vcenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
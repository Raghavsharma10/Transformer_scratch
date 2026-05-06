def get_vnetwork_dvpgs_input_vcenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        input = ET.SubElement(get_vnetwork_dvpgs, "input")
        vcenter = ET.SubElement(input, "vcenter")
        vcenter.text = kwargs.pop('vcenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
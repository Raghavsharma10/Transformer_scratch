def get_vnetwork_dvpgs_input_datacenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        input = ET.SubElement(get_vnetwork_dvpgs, "input")
        datacenter = ET.SubElement(input, "datacenter")
        datacenter.text = kwargs.pop('datacenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
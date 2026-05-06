def get_vnetwork_dvpgs_input_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        input = ET.SubElement(get_vnetwork_dvpgs, "input")
        name = ET.SubElement(input, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
def get_vnetwork_dvs_input_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvs = ET.Element("get_vnetwork_dvs")
        config = get_vnetwork_dvs
        input = ET.SubElement(get_vnetwork_dvs, "input")
        name = ET.SubElement(input, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
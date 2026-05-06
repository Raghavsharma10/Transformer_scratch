def get_vnetwork_dvs_input_datacenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvs = ET.Element("get_vnetwork_dvs")
        config = get_vnetwork_dvs
        input = ET.SubElement(get_vnetwork_dvs, "input")
        datacenter = ET.SubElement(input, "datacenter")
        datacenter.text = kwargs.pop('datacenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
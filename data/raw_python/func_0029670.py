def get_vnetwork_dvs_output_vnetwork_dvs_host(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvs = ET.Element("get_vnetwork_dvs")
        config = get_vnetwork_dvs
        output = ET.SubElement(get_vnetwork_dvs, "output")
        vnetwork_dvs = ET.SubElement(output, "vnetwork-dvs")
        host = ET.SubElement(vnetwork_dvs, "host")
        host.text = kwargs.pop('host')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
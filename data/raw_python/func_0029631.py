def get_vnetwork_hosts_input_vcenter(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_hosts = ET.Element("get_vnetwork_hosts")
        config = get_vnetwork_hosts
        input = ET.SubElement(get_vnetwork_hosts, "input")
        vcenter = ET.SubElement(input, "vcenter")
        vcenter.text = kwargs.pop('vcenter')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
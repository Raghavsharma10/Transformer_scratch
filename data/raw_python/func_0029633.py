def get_vnetwork_hosts_input_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_hosts = ET.Element("get_vnetwork_hosts")
        config = get_vnetwork_hosts
        input = ET.SubElement(get_vnetwork_hosts, "input")
        name = ET.SubElement(input, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
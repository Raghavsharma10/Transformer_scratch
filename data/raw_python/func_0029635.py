def get_vnetwork_hosts_output_vnetwork_hosts_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_hosts = ET.Element("get_vnetwork_hosts")
        config = get_vnetwork_hosts
        output = ET.SubElement(get_vnetwork_hosts, "output")
        vnetwork_hosts = ET.SubElement(output, "vnetwork-hosts")
        name = ET.SubElement(vnetwork_hosts, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
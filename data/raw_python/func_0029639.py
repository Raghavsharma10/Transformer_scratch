def get_vnetwork_hosts_output_vnetwork_hosts_vswitch(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_hosts = ET.Element("get_vnetwork_hosts")
        config = get_vnetwork_hosts
        output = ET.SubElement(get_vnetwork_hosts, "output")
        vnetwork_hosts = ET.SubElement(output, "vnetwork-hosts")
        vswitch = ET.SubElement(vnetwork_hosts, "vswitch")
        vswitch.text = kwargs.pop('vswitch')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
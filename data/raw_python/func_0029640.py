def get_vnetwork_hosts_output_vnetwork_hosts_interface_type(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_hosts = ET.Element("get_vnetwork_hosts")
        config = get_vnetwork_hosts
        output = ET.SubElement(get_vnetwork_hosts, "output")
        vnetwork_hosts = ET.SubElement(output, "vnetwork-hosts")
        interface_type = ET.SubElement(vnetwork_hosts, "interface-type")
        interface_type.text = kwargs.pop('interface_type')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
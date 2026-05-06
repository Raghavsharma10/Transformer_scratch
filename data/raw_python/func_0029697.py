def get_vnetwork_portgroups_output_vnetwork_pgs_host_nn(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_portgroups = ET.Element("get_vnetwork_portgroups")
        config = get_vnetwork_portgroups
        output = ET.SubElement(get_vnetwork_portgroups, "output")
        vnetwork_pgs = ET.SubElement(output, "vnetwork-pgs")
        host_nn = ET.SubElement(vnetwork_pgs, "host-nn")
        host_nn.text = kwargs.pop('host_nn')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
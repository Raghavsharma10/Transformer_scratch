def get_vnetwork_portgroups_output_vnetwork_pgs_vs_nn(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_portgroups = ET.Element("get_vnetwork_portgroups")
        config = get_vnetwork_portgroups
        output = ET.SubElement(get_vnetwork_portgroups, "output")
        vnetwork_pgs = ET.SubElement(output, "vnetwork-pgs")
        vs_nn = ET.SubElement(vnetwork_pgs, "vs-nn")
        vs_nn.text = kwargs.pop('vs_nn')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
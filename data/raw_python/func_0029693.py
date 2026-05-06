def get_vnetwork_portgroups_output_vnetwork_pgs_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_portgroups = ET.Element("get_vnetwork_portgroups")
        config = get_vnetwork_portgroups
        output = ET.SubElement(get_vnetwork_portgroups, "output")
        vnetwork_pgs = ET.SubElement(output, "vnetwork-pgs")
        name = ET.SubElement(vnetwork_pgs, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
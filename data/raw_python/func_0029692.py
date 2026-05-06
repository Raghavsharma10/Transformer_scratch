def get_vnetwork_portgroups_input_last_rcvd_instance(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_portgroups = ET.Element("get_vnetwork_portgroups")
        config = get_vnetwork_portgroups
        input = ET.SubElement(get_vnetwork_portgroups, "input")
        last_rcvd_instance = ET.SubElement(input, "last-rcvd-instance")
        last_rcvd_instance.text = kwargs.pop('last_rcvd_instance')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
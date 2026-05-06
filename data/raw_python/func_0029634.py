def get_vnetwork_hosts_input_last_rcvd_instance(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_hosts = ET.Element("get_vnetwork_hosts")
        config = get_vnetwork_hosts
        input = ET.SubElement(get_vnetwork_hosts, "input")
        last_rcvd_instance = ET.SubElement(input, "last-rcvd-instance")
        last_rcvd_instance.text = kwargs.pop('last_rcvd_instance')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
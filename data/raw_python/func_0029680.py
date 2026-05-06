def get_vnetwork_vswitches_input_last_rcvd_instance(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vswitches = ET.Element("get_vnetwork_vswitches")
        config = get_vnetwork_vswitches
        input = ET.SubElement(get_vnetwork_vswitches, "input")
        last_rcvd_instance = ET.SubElement(input, "last-rcvd-instance")
        last_rcvd_instance.text = kwargs.pop('last_rcvd_instance')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
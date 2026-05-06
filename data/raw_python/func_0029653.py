def get_vnetwork_vms_output_has_more(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        has_more = ET.SubElement(output, "has-more")
        has_more.text = kwargs.pop('has_more')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
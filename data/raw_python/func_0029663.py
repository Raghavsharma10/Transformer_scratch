def get_vnetwork_dvpgs_output_has_more(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        output = ET.SubElement(get_vnetwork_dvpgs, "output")
        has_more = ET.SubElement(output, "has-more")
        has_more.text = kwargs.pop('has_more')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
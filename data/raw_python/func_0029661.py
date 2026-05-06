def get_vnetwork_dvpgs_output_vnetwork_dvpgs_dvs_nn(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvpgs = ET.Element("get_vnetwork_dvpgs")
        config = get_vnetwork_dvpgs
        output = ET.SubElement(get_vnetwork_dvpgs, "output")
        vnetwork_dvpgs = ET.SubElement(output, "vnetwork-dvpgs")
        dvs_nn = ET.SubElement(vnetwork_dvpgs, "dvs-nn")
        dvs_nn.text = kwargs.pop('dvs_nn')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
def get_vnetwork_dvs_output_instance_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_dvs = ET.Element("get_vnetwork_dvs")
        config = get_vnetwork_dvs
        output = ET.SubElement(get_vnetwork_dvs, "output")
        instance_id = ET.SubElement(output, "instance-id")
        instance_id.text = kwargs.pop('instance_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
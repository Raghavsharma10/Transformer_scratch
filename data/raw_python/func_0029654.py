def get_vnetwork_vms_output_instance_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_vnetwork_vms = ET.Element("get_vnetwork_vms")
        config = get_vnetwork_vms
        output = ET.SubElement(get_vnetwork_vms, "output")
        instance_id = ET.SubElement(output, "instance-id")
        instance_id.text = kwargs.pop('instance_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
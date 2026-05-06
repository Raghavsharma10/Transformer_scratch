def vcenter_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        vcenter = ET.SubElement(config, "vcenter", xmlns="urn:brocade.com:mgmt:brocade-vswitch")
        id = ET.SubElement(vcenter, "id")
        id.text = kwargs.pop('id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
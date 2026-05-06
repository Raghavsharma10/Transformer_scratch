def vcenter_activate(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        vcenter = ET.SubElement(config, "vcenter", xmlns="urn:brocade.com:mgmt:brocade-vswitch")
        id_key = ET.SubElement(vcenter, "id")
        id_key.text = kwargs.pop('id')
        activate = ET.SubElement(vcenter, "activate")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
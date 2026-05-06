def vcenter_credentials_password(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        vcenter = ET.SubElement(config, "vcenter", xmlns="urn:brocade.com:mgmt:brocade-vswitch")
        id_key = ET.SubElement(vcenter, "id")
        id_key.text = kwargs.pop('id')
        credentials = ET.SubElement(vcenter, "credentials")
        password = ET.SubElement(credentials, "password")
        password.text = kwargs.pop('password')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
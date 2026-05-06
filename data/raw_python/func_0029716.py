def vcenter_credentials_vrf_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        vcenter = ET.SubElement(config, "vcenter", xmlns="urn:brocade.com:mgmt:brocade-vswitch")
        id_key = ET.SubElement(vcenter, "id")
        id_key.text = kwargs.pop('id')
        credentials = ET.SubElement(vcenter, "credentials")
        vrf_name = ET.SubElement(credentials, "vrf-name")
        vrf_name.text = kwargs.pop('vrf_name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
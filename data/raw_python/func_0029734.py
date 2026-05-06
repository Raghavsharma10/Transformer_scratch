def mac_access_list_standard_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac = ET.SubElement(config, "mac", xmlns="urn:brocade.com:mgmt:brocade-mac-access-list")
        access_list = ET.SubElement(mac, "access-list")
        standard = ET.SubElement(access_list, "standard")
        name = ET.SubElement(standard, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
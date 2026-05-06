def mac_access_list_extended_name(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        mac = ET.SubElement(config, "mac", xmlns="urn:brocade.com:mgmt:brocade-mac-access-list")
        access_list = ET.SubElement(mac, "access-list")
        extended = ET.SubElement(access_list, "extended")
        name = ET.SubElement(extended, "name")
        name.text = kwargs.pop('name')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
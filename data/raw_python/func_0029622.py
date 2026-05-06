def ntp_authentication_key_md5(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        ntp = ET.SubElement(config, "ntp", xmlns="urn:brocade.com:mgmt:brocade-ntp")
        authentication_key = ET.SubElement(ntp, "authentication-key")
        keyid_key = ET.SubElement(authentication_key, "keyid")
        keyid_key.text = kwargs.pop('keyid')
        md5 = ET.SubElement(authentication_key, "md5")
        md5.text = kwargs.pop('md5')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
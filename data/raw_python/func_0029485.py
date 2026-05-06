def ntp_server_key(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        ntp = ET.SubElement(config, "ntp", xmlns="urn:brocade.com:mgmt:brocade-ntp")
        server = ET.SubElement(ntp, "server")
        ip_key = ET.SubElement(server, "ip")
        ip_key.text = kwargs.pop('ip')
        use_vrf_key = ET.SubElement(server, "use-vrf")
        use_vrf_key.text = kwargs.pop('use_vrf')
        key = ET.SubElement(server, "key")
        key.text = kwargs.pop('key')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
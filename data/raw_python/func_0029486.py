def ntp_server_use_vrf(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        ntp = ET.SubElement(config, "ntp", xmlns="urn:brocade.com:mgmt:brocade-ntp")
        server = ET.SubElement(ntp, "server")
        ip_key = ET.SubElement(server, "ip")
        ip_key.text = kwargs.pop('ip')
        use_vrf = ET.SubElement(server, "use-vrf")
        use_vrf.text = kwargs.pop('use_vrf')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
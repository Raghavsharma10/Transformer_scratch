def ntp_server_ip(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        ntp = ET.SubElement(config, "ntp", xmlns="urn:brocade.com:mgmt:brocade-ntp")
        server = ET.SubElement(ntp, "server")
        use_vrf_key = ET.SubElement(server, "use-vrf")
        use_vrf_key.text = kwargs.pop('use_vrf')
        ip = ET.SubElement(server, "ip")
        ip.text = kwargs.pop('ip')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
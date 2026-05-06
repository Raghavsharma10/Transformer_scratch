def igmp_snooping_ip_pim_snooping_pimv4_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        igmp_snooping = ET.SubElement(config, "igmp-snooping", xmlns="urn:brocade.com:mgmt:brocade-igmp-snooping")
        ip = ET.SubElement(igmp_snooping, "ip")
        pim = ET.SubElement(ip, "pim")
        snooping = ET.SubElement(pim, "snooping")
        pimv4_enable = ET.SubElement(snooping, "pimv4-enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)
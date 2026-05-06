def get_stats(self):
        """Given XML version, parse create XMLAbstract object and sets xml_stats attribute."""
        self.gather_xml()
        self.xml_version = self.bs_xml.find('statistics')['version']

        if self.xml_version is None:
            raise XmlError("Unable to determine XML version via 'statistics' tag.")

        if self.xml_version == '3.6':
            self.stats = XmlV36(self.bs_xml)
        elif self.xml_version == '3.8':
            # 3.8 uses the same XML scheme as 3.6
            self.stats = XmlV36(self.bs_xml)
        elif self.xml_version == '3.11':
            # BIND 9.12 uses same XML schema as XmlV36
            self.stats = XmlV36(self.bs_xml)
        else:
            raise XmlError('Support must be added before being able to support newly-encountered XML version %s.' % self.xml_version)
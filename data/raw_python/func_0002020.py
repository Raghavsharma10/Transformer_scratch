def parse(self, fn):
        """
        Parses a file into a lxml.etree structure with namespaces remove.  This tree is added to self.iocs.

        :param fn: File to parse.
        :return:
        """
        ioc_xml = xmlutils.read_xml_no_ns(fn)
        if not ioc_xml:
            return False
        root = ioc_xml.getroot()
        iocid = root.get('id', None)
        if not iocid:
            return False
        self.iocs[iocid] = ioc_xml
        return True
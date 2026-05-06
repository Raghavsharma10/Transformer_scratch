def gather_xml(self):
        """Attempt to read the XML, whether from a file on-disk or via host:port.

        TODO: handle when you cant gather stats, due to bad hostname
        """
        if self.xml_filepath:
            with open(self.xml_filepath, "r") as xml_fh:
                self.raw_xml = xml_fh.read()
            self.bs_xml = BeautifulSoup(self.raw_xml, 'lxml')
        else:
            try:
                req = urlopen('http://%s:%s' % (self.host, self.port))
                self.raw_xml = req.read()
                self.bs_xml = BeautifulSoup(self.raw_xml, 'lxml')
            except URLError as u_error:
                raise XmlError('Unable to query BIND (%s:%s) for statistics. Reason: %s.' %
                               (self.host, self.port, u_error))
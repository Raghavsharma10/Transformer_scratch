def load_xml_conf(self, xml_file, id):
        '''
        Creates a new config from xml file.
        :param xml_file: path to xml file. Format : nutch-site.xml or nutch-default.xml
        :param id:
        :return: config object
        '''

        # converting nutch-site.xml to key:value pairs
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        params = {}
        for prop in tree.getroot().findall(".//property"):
            params[prop.find('./name').text.strip()] = prop.find('./value').text.strip()
        return self.proxy.Configs().create(id, configData=params)
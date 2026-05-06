def download_data(self, mock_response=None):
        """
        Loads XML data into the `xml_data` attribute.
        """
        if mock_response is not None:
            body = mock_response
        else:
            api_url = self.get_api_url()
            body = urlopen(api_url).read()
        xml_root = ElementTree.fromstring(body)
        xml_warnings = xml_root.find('warnings')
        if len(xml_warnings.attrib) != 0:
            print("Data warnings found: %s" % xml_warnings.attrib)
        xml_errors = xml_root.find('errors')
        if len(xml_errors.attrib) != 0:
            raise Exception("Data errors found: %s" % xml_errors.attrib)
        self.xml_data = xml_root.find('data')
def update_xml_element(self):
        """
        Updates the xml element contents to matches the instance contents.

        :returns: Updated XML element.
        :rtype: lxml.etree._Element
        """

        if not hasattr(self, 'xml_element'):
            self.xml_element = etree.Element(self.name, nsmap=NSMAP)

        if hasattr(self, 'lang'):
            self.xml_element.set(
                '{http://www.w3.org/XML/1998/namespace}lang', self.lang)
        if hasattr(self, 'override'):
            self.xml_element.set('override', str(self.override))
        if hasattr(self, 'text'):
            self.xml_element.text = self.text

        return self.xml_element
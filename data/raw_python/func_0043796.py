def update_xml_element(self):
        """
        Updates the xml element contents to matches the instance contents.

        :returns: Updated XML element.
        :rtype: lxml.etree._Element
        """

        if not hasattr(self, 'xml_element'):
            self.xml_element = etree.Element(self.name, nsmap=NSMAP)

        for element in self.xml_element:
            self.xml_element.remove(element)

        self.xml_element.tail = ''
        self.xml_element.text = self.convert_html_to_xml()

        return self.xml_element
def update_xml_element(self):
        """
        Updates the xml element contents to matches the instance contents.

        :returns: Updated XML element.
        :rtype: lxml.etree._Element
        """

        if not hasattr(self, 'xml_element'):
            self.xml_element = etree.Element(self.name, nsmap=NSMAP)

        self.xml_element.clear()

        if hasattr(self, 'abstract'):
            self.xml_element.set('abstract', self.abstract)
        if hasattr(self, 'prohibitChanges'):
            self.xml_element.set('prohibitChanges', self.prohibitChanges)
        self.xml_element.set('id', self.id)

        for child in self.children:
            if hasattr(child, 'update_xml_element'):
                child.update_xml_element()
                if hasattr(child, 'xml_element'):
                    self.xml_element.append(child.xml_element)

        return self.xml_element
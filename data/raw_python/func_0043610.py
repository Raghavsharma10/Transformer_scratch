def update_xml_element(self):
        """
        Updates the XML element contents to matches the instance contents.

        :returns: Updated XML element.
        :rtype: lxml.etree._Element
        """

        if not hasattr(self, 'xml_element'):
            self.xml_element = etree.Element(self.name, nsmap=NSMAP)

        self.xml_element.clear()

        if hasattr(self, 'resolved'):
            self.xml_element.set('resolved', self.resolved)
        if hasattr(self, 'style'):
            self.xml_element.set('style', self.style)
        if hasattr(self, 'style_href'):
            self.xml_element.set('style-href', self.style_href)
        if hasattr(self, 'lang'):
            self.xml_element.set(
                '{http://www.w3.org/XML/1998/namespace}lang', self.lang)
        self.xml_element.set('id', self.id)

        for child in self.children:
            if hasattr(child, 'update_xml_element'):
                child.update_xml_element()
                if hasattr(child, 'xml_element'):
                    self.xml_element.append(child.xml_element)

        return self.xml_element
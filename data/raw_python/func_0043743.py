def update_xml_element(self):
        """
        Updates the xml element contents to matches the instance contents.

        :returns: Updated XML element.
        :rtype: lxml.etree._Element
        """

        super(Description, self).update_xml_element()

        if hasattr(self, 'lang'):
            self.xml_element.set(
                '{http://www.w3.org/XML/1998/namespace}lang', self.lang)
        if hasattr(self, 'override'):
            self.xml_element.set('override', str(self.override))

        return self.xml_element
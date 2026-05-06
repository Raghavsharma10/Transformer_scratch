def to_xml_string(self):
        """
        Exports the element in XML format.

        :returns: element in XML format.
        :rtype: str
        """

        self.update_xml_element()
        xml = self.xml_element

        return etree.tostring(xml, pretty_print=True).decode('utf-8')
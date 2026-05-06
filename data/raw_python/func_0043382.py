def import_element(self, xml_element):
        """
        Imports the element from an lxml element and loads its content.

        :param lxml.etree._Element xml_element: XML element to import.
        """

        self.xml_element = xml_element

        uri, tag = Element.get_namespace_and_tag(self.xml_element.tag)
        self.namespace = uri
        self.name = tag

        self.load_xml_attrs()

        if self.xml_element.text is None:
            self.text = ''
        else:
            self.text = self.xml_element.text
def import_element(self, xml_element):
        """
        Imports the element from an lxml element and loads its content.
        """

        super(HTMLElement, self).import_element(xml_element)

        self.content = self.get_html_content()
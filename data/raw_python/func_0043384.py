def load_xml_attrs(self):
        """
        Load XML attributes as object attributes.

        :returns: List of parsed attributes.
        :rtype: list
        """

        attrs_list = list()

        if hasattr(self, 'xml_element'):
            xml_attrs = self.xml_element.attrib

            for variable, value in iter(xml_attrs.items()):
                uri, tag = Element.get_namespace_and_tag(variable)
                tag = tag.replace('-', '_')
                attrs_list.append(tag)
                setattr(self, tag, value)

            self.attrs = attrs_list

        return self.attrs
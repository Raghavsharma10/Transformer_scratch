def encode(self):
        """
        Encodes the object to a xml.etree.ElementTree.Element

        :return: the encoded element
        :rtype: xml.etree.ElementTree.Element
        """
        root_element = ElementTree.Element(self.TAG_NAME)
        for value in [value for value in self.__dict__.values() if isinstance(value, fields.Field)]:
            if value.required or value.value:
                root_element.append(value.encode())
        return root_element
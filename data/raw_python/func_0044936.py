def encode(self):
        """
        Just iterate over the child elements and append them to the current element

        :return: the encoded element
        :rtype: xml.etree.ElementTree.Element
        """
        element = ElementTree.Element(
            self.name,
            attrib={'type': FieldConstants.ARRAY},
        )
        for item in self.value:
            element.append(item.encode())
        return element
def encode(self):
        """
        Encodes the value of the field and put it in the element
        also make the check for nil=true if there is one

        :return: returns the encoded element
        :rtype: xml.etree.ElementTree.Element
        """
        element = ElementTree.Element(self.name)
        element = self._set_nil(element, lambda value: str(value))
        return element
def _set_nil(self, element, value_parser):
        """
        Method to set an attribute of the element.
        If the value of the field is None then set the nil='true' attribute in the element

        :param element: the element which needs to be modified
        :type element: xml.etree.ElementTree.Element
        :param value_parser: the lambda function which changes will be done to the self.value
        :type value_parser: def
        :return: the element with or without the specific attribute
        :rtype: xml.etree.ElementTree.Element
        """
        if self.value:
            element.text = value_parser(self.value)
        else:
            element.attrib['nil'] = 'true'
        return element
def decode(cls, root_element):
        """
        Decode the object to the object

        :param root_element: the parsed xml Element
        :type root_element: xml.etree.ElementTree.Element
        :return: the decoded Element as object
        :rtype: object
        """
        new_object = cls()
        field_names_to_attributes = new_object._get_field_names_to_attributes()
        for child_element in root_element:
            new_object._set_field(new_object, field_names_to_attributes, child_element)
        return new_object
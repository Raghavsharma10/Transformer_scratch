def set_attributes(self, attr_obj=None, ns_uri=None, **attr_dict):
        """
        Add or update this element's attributes, where attributes can be
        specified in a number of ways.

        :param attr_obj: a dictionary or list of attribute name/value pairs.
        :type attr_obj: dict, list, tuple, or None
        :param ns_uri: a URI defining a namespace for the new attributes.
        :type ns_uri: string or None
        :param dict attr_dict: attribute name and values specified as keyword
            arguments.
        """
        self._set_element_attributes(self.impl_node,
            attr_obj=attr_obj, ns_uri=ns_uri, **attr_dict)
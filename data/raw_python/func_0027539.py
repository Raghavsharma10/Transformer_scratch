def namespace_uri(self, name):
        """
        :param string name: the name of an attribute to look up.

        :return: the namespace URI associated with the named attribute,
            or None.
        """
        a_node = self.adapter.get_node_attribute_node(self.impl_element, name)
        if a_node is None:
            return None
        return self.adapter.get_node_namespace_uri(a_node)
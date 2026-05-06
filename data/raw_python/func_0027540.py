def prefix(self, name):
        """
        :param string name: the name of an attribute to look up.

        :return: the prefix component of the named attribute's name,
            or None.
        """
        a_node = self.adapter.get_node_attribute_node(self.impl_element, name)
        if a_node is None:
            return None
        return a_node.prefix
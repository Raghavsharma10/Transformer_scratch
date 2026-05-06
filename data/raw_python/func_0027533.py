def attribute_node(self, name, ns_uri=None):
        """
        :param string name: the name of the attribute to return.
        :param ns_uri: a URI defining a namespace constraint on the attribute.
        :type ns_uri: string or None

        :return: this element's attributes that match ``ns_uri`` as
            :class:`Attribute` nodes.
        """
        attr_impl_node = self.adapter.get_node_attribute_node(
            self.impl_node, name, ns_uri)
        return self.adapter.wrap_node(
            attr_impl_node, self.adapter.impl_document, self.adapter)
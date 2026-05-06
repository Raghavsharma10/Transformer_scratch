def attribute_nodes(self):
        """
        :return: a list of this element's attributes as
            :class:`Attribute` nodes.
        """
        impl_attr_nodes = self.adapter.get_node_attributes(self.impl_node)
        wrapped_attr_nodes = [
            self.adapter.wrap_node(a, self.adapter.impl_document, self.adapter)
            for a in impl_attr_nodes]
        return sorted(wrapped_attr_nodes, key=lambda x: x.name)
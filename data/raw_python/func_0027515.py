def parent(self):
        """
        :return: the parent of this node, or *None* of the node has no parent.
        """
        parent_impl_node = self.adapter.get_node_parent(self.impl_node)
        return self.adapter.wrap_node(
            parent_impl_node, self.adapter.impl_document, self.adapter)
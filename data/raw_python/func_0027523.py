def transplant_node(self, node):
        """
        Transplant a node from another document to become a child of this node,
        removing it from the source document.  The node to be transplanted can
        be a :class:`Node` based on the same underlying XML library
        implementation and adapter, or a "raw" node from that implementation.

        :param node: the node in another document to transplant.
        :type node: xml4h or implementation node
        """
        if isinstance(node, xml4h.nodes.Node):
            child_impl_node = node.impl_node
            original_parent_impl_node = node.parent.impl_node
        else:
            child_impl_node = node  # Assume it's a valid impl node
            original_parent_impl_node = self.adapter.get_node_parent(node)
        self.adapter.import_node(self.impl_node, child_impl_node,
            original_parent_impl_node, clone=False)
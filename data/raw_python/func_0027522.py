def clone_node(self, node):
        """
        Clone a node from another document to become a child of this node, by
        copying the node's data into this document but leaving the node
        untouched in the source document. The node to be cloned can be
        a :class:`Node` based on the same underlying XML library implementation
        and adapter, or a "raw" node from that implementation.

        :param node: the node in another document to clone.
        :type node: xml4h or implementation node
        """
        if isinstance(node, xml4h.nodes.Node):
            child_impl_node = node.impl_node
        else:
            child_impl_node = node  # Assume it's a valid impl node
        self.adapter.import_node(self.impl_node, child_impl_node, clone=True)
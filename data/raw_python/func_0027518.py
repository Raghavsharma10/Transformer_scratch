def siblings(self):
        """
        :return: a list of this node's sibling nodes.
        :rtype: NodeList
        """
        impl_nodelist = self.adapter.get_node_children(self.parent.impl_node)
        return self._convert_nodelist(
            [n for n in impl_nodelist if n != self.impl_node])
def children(self):
        """
        :return: a :class:`NodeList` of this node's child nodes.
        """
        impl_nodelist = self.adapter.get_node_children(self.impl_node)
        return self._convert_nodelist(impl_nodelist)
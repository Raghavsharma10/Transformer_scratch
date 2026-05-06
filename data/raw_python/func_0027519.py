def siblings_before(self):
        """
        :return: a list of this node's siblings that occur *before* this
            node in the DOM.
        """
        impl_nodelist = self.adapter.get_node_children(self.parent.impl_node)
        before_nodelist = []
        for n in impl_nodelist:
            if n == self.impl_node:
                break
            before_nodelist.append(n)
        return self._convert_nodelist(before_nodelist)
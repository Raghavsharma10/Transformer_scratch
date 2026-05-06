def siblings_after(self):
        """
        :return: a list of this node's siblings that occur *after* this
            node in the DOM.
        """
        impl_nodelist = self.adapter.get_node_children(self.parent.impl_node)
        after_nodelist = []
        is_after_myself = False
        for n in impl_nodelist:
            if is_after_myself:
                after_nodelist.append(n)
            elif n == self.impl_node:
                is_after_myself = True
        return self._convert_nodelist(after_nodelist)
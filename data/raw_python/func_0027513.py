def root(self):
        """
        :return: the root :class:`Element` node of the document that
            contains this node, or ``self`` if this node is the root element.
        """
        if self.is_root:
            return self
        return self.adapter.wrap_node(
            self.adapter.impl_root_element, self.adapter.impl_document,
            self.adapter)
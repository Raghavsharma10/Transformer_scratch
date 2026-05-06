def delete(self, destroy=True):
        """
        Delete this node from the owning document.

        :param bool destroy: if True the child node will be destroyed in
            addition to being removed from the document.

        :returns: the removed child node, or *None* if the child was destroyed.
        """
        removed_child = self.adapter.remove_node_child(
            self.adapter.get_node_parent(self.impl_node), self.impl_node,
            destroy_node=destroy)
        if removed_child is not None:
            return self.adapter.wrap_node(removed_child, None, self.adapter)
        else:
            return None
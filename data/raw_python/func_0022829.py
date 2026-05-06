def node_path_to_child(self, node):
        """Return a list describing the path from this node to a child node

        If *node* is not a (grand)child of this node, then raise RuntimeError.

        Parameters
        ----------
        node : instance of Node
            The child node.

        Returns
        -------
        path : list | None
            The path.
        """
        if node is self:
            return []

        # Go up from the child node as far as we can
        path1 = [node]
        child = node
        while child.parent is not None:
            child = child.parent
            path1.append(child)
            # Early exit
            if child is self:
                return list(reversed(path1))
        
        # Verify that we're not cut off
        if path1[-1].parent is None:
            raise RuntimeError('%r is not a child of %r' % (node, self))
        
        def _is_child(path, parent, child):
            path.append(parent)
            if child in parent.children:
                return path
            else:
                for c in parent.children:
                    possible_path = _is_child(path[:], c, child)
                    if possible_path:
                        return possible_path
            return None

        # Search from the parent towards the child
        path2 = _is_child([], self, path1[-1])
        if not path2:
            raise RuntimeError('%r is not a child of %r' % (node, self))

        # Return
        return path2 + list(reversed(path1))
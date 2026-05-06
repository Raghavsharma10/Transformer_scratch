def common_parent(self, node):
        """
        Return the common parent of two entities

        If the entities have no common parent, return None.

        Parameters
        ----------
        node : instance of Node
            The other node.

        Returns
        -------
        parent : instance of Node | None
            The parent.
        """
        p1 = self.parent_chain()
        p2 = node.parent_chain()
        for p in p1:
            if p in p2:
                return p
        return None
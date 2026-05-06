def collapse_nodes(self, min_dist=1e-6, min_support=0):
        """
        Returns a copy of the tree where internal nodes with dist <= min_dist
        are deleted, resulting in a collapsed tree. e.g.:

        newtre = tre.collapse_nodes(min_dist=0.001)
        newtre = tre.collapse_nodes(min_support=50)
        """
        nself = self.copy()
        for node in nself.treenode.traverse():
            if not node.is_leaf():
                if (node.dist <= min_dist) | (node.support < min_support):
                    node.delete()
        nself._coords.update()
        return nself
def node_scale_root_height(self, treeheight=1):
        """
        Returns a toytree copy with all nodes scaled so that the root 
        height equals the value entered for treeheight.
        """
        # make tree height = 1 * treeheight
        ctree = self._ttree.copy()
        _height = ctree.treenode.height
        for node in ctree.treenode.traverse():
            node.dist = (node.dist / _height) * treeheight
        ctree._coords.update()
        return ctree
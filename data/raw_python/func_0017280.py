def node_multiplier(self, multiplier=0.5, seed=None):
        """
        Returns a toytree copy with all nodes multiplied by a constant 
        sampled uniformly between (multiplier, 1/multiplier).
        """
        random.seed(seed)
        ctree = self._ttree.copy()
        low, high = sorted([multiplier, 1. / multiplier])
        mult = random.uniform(low, high)
        for node in ctree.treenode.traverse():
            node.dist = node.dist * mult
        ctree._coords.update()
        return ctree
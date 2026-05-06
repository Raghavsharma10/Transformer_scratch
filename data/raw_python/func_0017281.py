def make_ultrametric(self, strategy=1):
        """
        Returns a tree with branch lengths transformed so that the tree is 
        ultrametric. Strategies include (1) tip-align: extend tips to the length
        of the fartest tip from the root; (2) non-parametric rate-smoothing: 
        minimize ancestor-descendant local rates on branches to align tips (
        not yet supported); and (3) penalized-likelihood: not yet supported.
        """
        ctree = self._ttree.copy()

        if strategy == 1:
            for node in ctree.treenode.traverse():
                if node.is_leaf():
                    node.dist = node.height + 1

        else:
            raise NotImplementedError(
                "Strategy {} not yet implemented. Seeking developers."
                .format(strategy))

        return ctree
def unittree(ntips, treeheight=1.0, seed=None):
        """
        Returns a random tree topology w/ N tips and a root height set to
        1 or a user-entered treeheight value. Descendant nodes are evenly 
        spaced between the root and time 0.

        Parameters
        -----------
        ntips (int):
            The number of tips in the randomly generated tree

        treeheight(float):
            Scale tree height (all edges) so that root is at this height.

        seed (int):
            Random number generator seed.
        """
        # seed generator
        random.seed(seed)

        # generate tree with N tips.
        tmptree = TreeNode()
        tmptree.populate(ntips)
        self = toytree.tree(newick=tmptree.write())

        # set tip names by labeling sequentially from 0
        self = (
            self
            .ladderize()
            .mod.make_ultrametric()
            .mod.node_scale_root_height(treeheight)
        )

        # set tipnames randomly (doesn't have to match idx)
        nidx = list(range(self.ntips))
        random.shuffle(nidx)
        for tidx, node in enumerate(self.treenode.get_leaves()):
            node.name = "r{}".format(nidx[tidx])

        # fill internal node names and idx
        self._coords.update()
        return self
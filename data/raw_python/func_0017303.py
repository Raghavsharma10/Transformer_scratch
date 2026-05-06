def root(self, names=None, wildcard=None, regex=None):
        """
        (Re-)root a tree by creating selecting a existing split in the tree,
        or creating a new node to split an edge in the tree. Rooting location
        is selected by entering the tips descendant from one child of the root
        split (e.g., names='a' or names=['a', 'b']). You can alternatively
        select a list of tip names using a fuzzy selector based on a unique
        shared string (wildcard="prz") or a regex matching pattern.

        Example:
        To root on a clade that includes the samples "1-A" and "1-B" you can
        do any of the following:

        rtre = tre.root(outgroup=["1-A", "1-B"])
        rtre = tre.root(wildcard="1-")
        rtre = tre.root(regex="1-[A,B]")
        """
        # make a deepcopy of the tree
        nself = self.copy()

        # get treenode of the common ancestor of selected tips
        try:
            node = fuzzy_match_tipnames(
                nself, names, wildcard, regex, True, True)       

        except ToytreeError:           
            # try reciprocal taxon list
            tipnames = fuzzy_match_tipnames(
                nself, names, wildcard, regex, False, False)
            tipnames = list(set(self.get_tip_labels()) - set(node))
            node = fuzzy_match_tipnames(
                nself, tipnames, None, None, True, True)

        # split root node if more than di- as this is the unrooted state
        if not nself.is_bifurcating():
            nself.treenode.resolve_polytomy()

        # root the object with ete's translate
        nself.treenode.set_outgroup(node)
        nself._coords.update()

        # get features
        testnode = nself.treenode.get_leaves()[0]
        features = {"name", "dist", "support", "height"}
        extrafeat = {i for i in testnode.features if i not in features}
        features.update(extrafeat)

        # if there is a new node now, clean up its features
        nnode = [i for i in nself.treenode.traverse() if not hasattr(i, "idx")]
        if nnode:
            # nnode is the node that was added
            # rnode is the location where it *should* have been added
            nnode = nnode[0]
            rnode = [i for i in nself.treenode.children if i != node][0]

            # get idxs of existing nodes
            idxs = [int(i.idx) for i in nself.treenode.traverse()
                    if hasattr(i, "idx")]

            # newnode is a tip
            if len(node.is_leaf()) == 1:
                nnode.name = str("rerooted")
                rnode.name = node
                rnode.add_feature("idx", max(idxs) + 1)
                rnode.dist *= 2
                sister = rnode.get_sisters()[0]
                sister.dist *= 2
                rnode.support = 100
                for feature in extrafeat:
                    nnode.add_feature(feature, getattr(rnode, feature))
                    rnode.del_feature(feature)

            # newnode is internal
            else:
                nnode.add_feature("idx", max(idxs) + 1)
                nnode.name = str("rerooted")
                nnode.dist *= 2
                sister = nnode.get_sisters()[0]
                sister.dist *= 2
                nnode.support = 100

        # store tree back into newick and reinit Toytree with new newick
        # if NHX format then preserve the NHX features.
        nself.treenode.ladderize()
        nself._coords.update()
        return nself
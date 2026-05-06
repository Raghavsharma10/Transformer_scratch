def rotate_node(
        self, 
        names=None, 
        wildcard=None, 
        regex=None, 
        idx=None, 
        # modify_tree=False,
        ):
        """
        Returns a ToyTree with the selected node rotated for plotting.
        tip colors do not align correct currently if nodes are rotated...
        """
        # make a copy
        revd = {j: i for (i, j) in enumerate(self.get_tip_labels())}
        neworder = {}

        # get node to rotate
        treenode = fuzzy_match_tipnames(
            self, names, wildcard, regex, True, True)
        children = treenode.up.children
        names = [[j.name for j in i.get_leaves()] for i in children]
        nidxs = [[revd[i] for i in j] for j in names]

        # get size of the big clade
        move = max((len(i) for i in nidxs))
        if len(nidxs[0]) > len(nidxs[1]):
            move = min((len(i) for i in nidxs))

        # newdict
        cnames = list(itertools.chain(*names))
        tdict = {i: None for i in cnames}
        cycle = itertools.cycle(itertools.chain(*nidxs))
        for m in range(move):
            next(cycle)
        for t in cnames:
            tdict[t] = next(cycle)

        for key in revd:
            if key in tdict:
                neworder[key] = tdict[key]
            else:
                neworder[key] = revd[key]

        revd = {j: i for (i, j) in neworder.items()}
        neworder = [revd[i] for i in range(self.ntips)]

        # returns a new tree (i.e., copy) modified w/ a fixed order
        nself = ToyTree(self.newick, fixed_order=neworder)
        nself._coords.update()
        return nself
def coaltree(ntips, ne=None, seed=None):
        """
        Returns a coalescent tree with ntips samples and waiting times 
        between coalescent events drawn from the kingman coalescent:
        (4N)/(k*(k-1)), where N is population size and k is sample size.
        Edge lengths on the tree are in generations.

        If no Ne argument is entered then edge lengths are returned in units
        of 2*Ne, i.e., coalescent time units. 
        """

        # seed generator
        random.seed(seed)

        # convert units
        coalunits = False
        if not ne:
            coalunits = True
            ne = 10000

        # build tree: generate N tips as separate Nodes then attach together 
        # at internal nodes drawn randomly from coalescent waiting times.
        tips = [
            toytree.tree().treenode.add_child(name=str(i)) 
            for i in range(ntips)
        ]
        while len(tips) > 1:
            rtree = toytree.tree()
            tip1 = tips.pop(random.choice(range(len(tips))))
            tip2 = tips.pop(random.choice(range(len(tips))))
            kingman = (4. * ne) / float(ntips * (ntips - 1))
            dist = random.expovariate(1. / kingman)
            rtree.treenode.add_child(tip1, dist=tip2.height + dist)
            rtree.treenode.add_child(tip2, dist=tip1.height + dist)
            tips.append(rtree.treenode)

        # build new tree from the newick string
        self = toytree.tree(tips[0].write())    
        self.treenode.ladderize()

        # make tree edges in units of 2N (then N doesn't matter!)
        if coalunits:
            for node in self.treenode.traverse():
                node.dist /= (2. * ne)

        # ensure tips are at zero (they sometime vary just slightly)
        for node in self.treenode.traverse():
            if node.is_leaf():
                node.dist += node.height

        # set tipnames
        for tip in self.get_tip_labels():
            node = self.treenode.search_nodes(name=tip)[0]
            node.name = "r{}".format(node.name)

        # decompose fills in internal node names and idx
        self._coords.update()
        return self
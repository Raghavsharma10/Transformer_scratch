def imbtree(ntips, treeheight=1.0):
        """
        Return an imbalanced (comb-like) tree topology.
        """
        rtree = toytree.tree()
        rtree.treenode.add_child(name="0")
        rtree.treenode.add_child(name="1")

        for i in range(2, ntips):
            # empty node
            cherry = toytree.tree()
            # add new child
            cherry.treenode.add_child(name=str(i))
            # add old tree
            cherry.treenode.add_child(rtree.treenode)
            # update rtree
            rtree = cherry
        
        # get toytree from newick            
        tre = toytree.tree(rtree.write(tree_format=9))
        tre = tre.mod.make_ultrametric()
        self = tre.mod.node_scale_root_height(treeheight)
        self._coords.update()
        return self
def baltree(ntips, treeheight=1.0):
        """
        Returns a balanced tree topology.
        """
        # require even number of tips
        if ntips % 2:
            raise ToytreeError("balanced trees must have even number of tips.")

        # make first cherry
        rtree = toytree.tree()
        rtree.treenode.add_child(name="0")
        rtree.treenode.add_child(name="1")

        # add tips in a balanced way
        for i in range(2, ntips):

            # get node to split
            node = return_small_clade(rtree.treenode)

            # add two children
            node.add_child(name=node.name)
            node.add_child(name=str(i))

            # rename ancestral node
            node.name = None

        # rename tips so names are in order
        idx = 0
        for node in rtree.treenode.traverse("postorder"):
            if node.is_leaf():
                node.name = str(idx)
                idx += 1
        
        # get toytree from newick            
        tre = toytree.tree(rtree.write(tree_format=9))
        tre = tre.mod.make_ultrametric()
        self = tre.mod.node_scale_root_height(treeheight)
        self._coords.update()
        return self
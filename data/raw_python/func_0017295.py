def get_tip_labels(self, idx=None):
        """
        Returns tip labels in the order they will be plotted on the tree, i.e.,
        starting from zero axis and counting up by units of 1 (bottom to top 
        in right-facing trees; left to right in down-facing). If 'idx' is 
        indicated then a list of tip labels descended from that node will be 
        returned, instead of all tip labels. This is useful in combination 
        with other functions that select nodes/clades of the tree based on a 
        list of tip labels. You can use the toytree draw() command with 
        tip_labels='idx' or tip_labels=True to see idx labels plotted on nodes. 

        Parameters:
            idx (int): index label of a node.

        Example:
            # select a clade of the tree and use it for rooting.
            tiplist = tre.get_descenants_from_idx(21)
            tre.root(names=tiplist)
        """
        if not idx:
            return self.treenode.get_leaf_names()[::-1]
        else:
            treenode = self.treenode.search_nodes(idx=idx)[0]
            return treenode.get_leaf_names()[::-1]
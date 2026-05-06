def fit_tip_labels(self):
        """
        Modifies display range to ensure tip labels fit. This is a bit hackish
        still. The problem is that the 'extents' range of the rendered text
        is totally correct. So we add a little buffer here. Should add for 
        user to be able to modify this if needed. If not using edge lengths
        then need to use unit length for treeheight.
        """

        if not self.tip_labels:
            return 

        # longest name (this will include html hacks)
        longest_name = max([len(i) for i in self.tip_labels])
        if longest_name > 10:
            multiplier = 0.85
        else:
            multiplier = 0.25

        if self.style.use_edge_lengths:
            addon = (self.treelist[0].treenode.height + (
                self.treelist[0].treenode.height * multiplier))
        else:
            addon = self.treelist[0].treenode.get_farthest_leaf(True)[1]

        # modify display for orientations
        if self.style.orient == "right":
            self.axes.x.domain.max = addon
        elif self.style.orient == "down":
            self.axes.y.domain.min = -1 * addon
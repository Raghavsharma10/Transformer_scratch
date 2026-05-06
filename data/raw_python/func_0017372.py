def fit_tip_labels(self):
        """
        Modifies display range to ensure tip labels fit. This is a bit hackish
        still. The problem is that the 'extents' range of the rendered text
        is totally correct. So we add a little buffer here. Should add for 
        user to be able to modify this if needed. If not using edge lengths
        then need to use unit length for treeheight.
        """
        # user entered values
        #if self.style.axes.x_domain_max or self.style.axes.y_domain_min:
        #    self.axes.x.domain.max = self.style.axes.x_domain_max
        #    self.axes.y.domain.min = self.style.axes.y_domain_min            

        # IF USE WANTS TO CHANGE IT THEN DO IT AFTER USING AXES
        # or auto-fit (tree height)
        #else:
        if self.style.use_edge_lengths:
            addon = self.ttree.treenode.height * .85
        else:
            addon = self.ttree.treenode.get_farthest_leaf(True)[1]

        # modify display for orientations
        if self.style.tip_labels:
            if self.style.orient == "right":
                self.axes.x.domain.max = addon
            elif self.style.orient == "down":
                self.axes.y.domain.min = -1 * addon
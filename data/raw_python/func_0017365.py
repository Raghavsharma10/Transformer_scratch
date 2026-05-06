def update_fixed_order(self):
        "after pruning fixed order needs update to match new nnodes/ntips."
        # set tips order if fixing for multi-tree plotting (default None)
        fixed_order = self.ttree._fixed_order
        self.ttree_fixed_order = None
        self.ttree_fixed_idx = list(range(self.ttree.ntips))

        # check if fixed_order changed:
        if fixed_order:
            fixed_order = [
                i for i in fixed_order if i in self.ttree.get_tip_labels()]
            self.ttree._set_fixed_order(fixed_order)
        else:
            self.ttree._fixed_idx = list(range(self.ttree.ntips))
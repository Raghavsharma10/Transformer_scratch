def update(self):
        "Updates cartesian coordinates for drawing tree graph"              
        # get new shape and clear for attrs
        self.edges = np.zeros((self.ttree.nnodes - 1, 2), dtype=int)
        self.verts = np.zeros((self.ttree.nnodes, 2), dtype=float)
        self.lines = []
        self.coords = []

        # fill with updates
        self.update_idxs()             # get dimensions of tree
        self.update_fixed_order()      # in case ntips changed
        self.assign_vertices()         # get node locations
        self.assign_coordinates()      # get edge locations
        self.reorient_coordinates()
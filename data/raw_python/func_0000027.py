def add_nodes(self, coors, node_low_or_high=None):
        """
        Add new nodes at the end of the list.
        """
        last = self.lastnode
        if type(coors) is nm.ndarray:
            if len(coors.shape) == 1:
                coors = coors.reshape((1, coors.size))

            nadd = coors.shape[0]
            idx = slice(last, last + nadd)
        else:
            nadd = 1
            idx = self.lastnode
        right_dimension = coors.shape[1]
        self.nodes[idx, :right_dimension] = coors
        self.node_flag[idx] = True
        self.lastnode += nadd
        self.nnodes += nadd
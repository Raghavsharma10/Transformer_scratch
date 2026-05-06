def get_midpoint_outgroup(self):
        """
        Returns the node that divides the current tree into two 
        distance-balanced partitions.
        """
        # Gets the farthest node to the current root
        root = self.get_tree_root()
        nA, r2A_dist = root.get_farthest_leaf()
        nB, A2B_dist = nA.get_farthest_node()

        outgroup = nA
        middist = A2B_dist / 2.0
        cdist = 0
        current = nA
        while current is not None:
            cdist += current.dist
            if cdist > (middist): # Deja de subir cuando se pasa del maximo
                break
            else:
                current = current.up
        return current
def clear(self):
        """Clears the Merkle Tree by releasing the Merkle root and each leaf's references, the rest
        should be garbage collected.  This may be useful for situations where you want to take an existing
        tree, make changes to the leaves, but leave it uncalculated for some time, without node
        references that are no longer correct still hanging around. Usually it is better just to make
        a new tree.
        """
        self.root = None
        for leaf in self.leaves:
            leaf.p, leaf.sib, leaf.side = (None, ) * 3
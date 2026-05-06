def add_leaf(self, leaf_blob):
        """Adds a leaf to the list of leaves.  Does not build the tree so call
        `build()` to construct the rest of the tree from the added leaves.

        :param leaf_blob: the leaf payload to add.  should be a hashable object
        """
        self.leaves.append(MerkleLeaf(len(self.leaves), leaf_blob))
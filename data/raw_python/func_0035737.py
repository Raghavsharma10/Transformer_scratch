def get_branch(self, i):
        """Gets a branch associated with leaf i.  This will trace the tree
        from the leaves down to the root, constructing a list of tuples that
        represent the pairs of nodes all the way from leaf i to the root.

        :param i: the leaf identifying the branch to retrieve
        """
        branch = MerkleBranch(self.order)
        j = i + 2 ** self.order - 1

        for k in range(0, self.order):
            if (self.is_left(j)):
                branch.set_row(k, (self.nodes[j], self.nodes[j + 1]))
            else:
                branch.set_row(k, (self.nodes[j - 1], self.nodes[j]))
            j = MerkleTree.get_parent(j)

        return branch
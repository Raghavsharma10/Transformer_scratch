def get_all_chains(self):
        """Assemble and return a list of all chains for all leaf nodes to the merkle root.
        """
        return [self.get_chain(i) for i in range(len(self.leaves))]
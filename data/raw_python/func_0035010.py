def tree(self):
        """Tree with branch lengths in codon substitutions per site.

        The tree is a `Bio.Phylo.BaseTree.Tree` object.

        This is the current tree after whatever optimizations have
        been performed so far.
        """
        bs = self.model.branchScale
        for node in self._tree.find_clades():
            if node != self._tree.root:
                node.branch_length = self.t[self.name_to_nodeindex[node]] * bs
        return self._tree
def get_consensus_tree(self, cutoff=0.0, best_tree=None):
        """
        Returns an extended majority rule consensus tree as a Toytree object.
        Node labels include 'support' values showing the occurrence of clades 
        in the consensus tree across trees in the input treelist. 
        Clades with support below 'cutoff' are collapsed into polytomies.
        If you enter an optional 'best_tree' then support values from
        the treelist calculated for clades in this tree, and the best_tree is
        returned with support values added to nodes. 

        Params
        ------
        cutoff (float; default=0.0): 
            Cutoff below which clades are collapsed in the majority rule 
            consensus tree. This is a proportion (e.g., 0.5 means 50%). 
        best_tree (Toytree; optional):
            A tree that support values should be calculated for and added to. 
            For example, you want to calculate how often clades in your best 
            ML tree are supported in 100 bootstrap trees. 
        """
        if best_tree:
            raise NotImplementedError("best_tree option not yet supported.")
        cons = ConsensusTree(self.treelist, cutoff)
        cons.update()
        return cons.ttree
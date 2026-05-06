def get_mrca_idx_from_tip_labels(self, names=None, wildcard=None, regex=None):
        """
        Returns the node idx label of the most recent common ancestor node 
        for the clade that includes the selected tips. Arguments can use fuzzy
        name matching: a list of tip names, wildcard selector, or regex string.
        """
        if not any([names, wildcard, regex]):
            raise ToytreeError("at least one argument required")
        node = fuzzy_match_tipnames(
            self, names, wildcard, regex, True, False)
        return node.idx
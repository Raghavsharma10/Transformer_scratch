def drop_tips(self, names=None, wildcard=None, regex=None):
        """
        Returns a copy of the tree with the selected tips removed. The entered
        value can be a name or list of names. To prune on an internal node to
        create a subtree see the .prune() function instead.

        Parameters:
        tips: list of tip names.

        # example:
        ptre = tre.drop_tips(['a', 'b'])
        """
        # make a deepcopy of the tree
        nself = self.copy()

        # return if nothing to drop
        if not any([names, wildcard, regex]):
            return nself

        # get matching names list with fuzzy match
        tipnames = fuzzy_match_tipnames(
            ttree=nself,
            names=names,
            wildcard=wildcard,
            regex=regex,
            mrca=False,
            mono=False,
        )

        if len(tipnames) == len(nself):
            raise ToytreeError("You cannot drop all tips from the tree.")

        if not tipnames:
            raise ToytreeError("No tips selected.")

        keeptips = [i for i in nself.get_tip_labels() if i not in tipnames]
        nself.treenode.prune(keeptips, preserve_branch_length=True)
        nself._coords.update()
        return nself
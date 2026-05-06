def check_monophyly(self, 
        values, 
        target_attr, 
        ignore_missing=False,
        unrooted=False):
        """
        Returns True if a given target attribute is monophyletic under
        this node for the provided set of values.

        If not all values are represented in the current tree
        structure, a ValueError exception will be raised to warn that
        strict monophyly could never be reached (this behaviour can be
        avoided by enabling the `ignore_missing` flag.
        
        Parameters:
        -----------
        values: 
            a set of values for which monophyly is expected.
        
        target_attr: 
            node attribute being used to check monophyly (i.e. species for 
            species trees, names for gene family trees, or any custom feature
            present in the tree).

        ignore_missing: 
            Avoid raising an Exception when missing attributes are found.

        unrooted: 
            If True, tree will be treated as unrooted, thus allowing to find
            monophyly even when current outgroup is spliting a monophyletic group.

        Returns: 
        --------
        the following tuple
        IsMonophyletic (boolean),
        clade type ('monophyletic', 'paraphyletic' or 'polyphyletic'),
        leaves breaking the monophyly (set)
        """
        if type(values) != set:

            values = set(values)

        # This is the only time I traverse the tree, then I use cached
        # leaf content
        n2leaves = self.get_cached_content()

        # Raise an error if requested attribute values are not even present
        if ignore_missing:
            found_values = set([getattr(n, target_attr) for n in n2leaves[self]])
            missing_values = values - found_values
            values = values & found_values

        # Locate leaves matching requested attribute values
        targets = set([leaf for leaf in n2leaves[self]
                   if getattr(leaf, target_attr) in values])
        if not ignore_missing:
            if values - set([getattr(leaf, target_attr) for leaf in targets]):
                raise ValueError('The monophyly of the provided values could never be reached, as not all of them exist in the tree.'
                                 ' Please check your target attribute and values, or set the ignore_missing flag to True')

        if unrooted:
            smallest = None
            for side1, side2 in self.iter_edges(cached_content=n2leaves):
                if targets.issubset(side1) and (not smallest or len(side1) < len(smallest)):
                    smallest = side1
                elif targets.issubset(side2) and (not smallest or len(side2) < len(smallest)):
                        smallest = side2
                if smallest is not None and len(smallest) == len(targets):
                    break
            foreign_leaves = smallest - targets
        else:
            # Check monophyly with get_common_ancestor. Note that this
            # step does not require traversing the tree again because
            # targets are node instances instead of node names, and
            # get_common_ancestor function is smart enough to detect it
            # and avoid unnecessary traversing.
            common = self.get_common_ancestor(targets)
            observed = n2leaves[common]
            foreign_leaves = set([leaf for leaf in observed
                              if getattr(leaf, target_attr) not in values])

        if not foreign_leaves:
            return True, "monophyletic", foreign_leaves
        else:
            # if the requested attribute is not monophyletic in this
            # node, let's differentiate between poly and paraphyly.
            poly_common = self.get_common_ancestor(foreign_leaves)
            # if the common ancestor of all foreign leaves is self
            # contained, we have a paraphyly. Otherwise, polyphyly.
            polyphyletic = [leaf for leaf in poly_common if
                            getattr(leaf, target_attr) in values]
            if polyphyletic:
                return False, "polyphyletic", foreign_leaves
            else:
                return False, "paraphyletic", foreign_leaves
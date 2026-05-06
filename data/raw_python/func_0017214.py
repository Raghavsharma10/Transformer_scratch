def get_monophyletic(self, values, target_attr):
        """
        Returns a list of nodes matching the provided monophyly
        criteria. For a node to be considered a match, all
        `target_attr` values within and node, and exclusively them,
        should be grouped.

        :param values: a set of values for which monophyly is
            expected.

        :param target_attr: node attribute being used to check
            monophyly (i.e. species for species trees, names for gene
            family trees).
        """
        if type(values) != set:
            values = set(values)

        n2values = self.get_cached_content(store_attr=target_attr)

        is_monophyletic = lambda node: n2values[node] == values
        for match in self.iter_leaves(is_leaf_fn=is_monophyletic):
            if is_monophyletic(match):
                yield match
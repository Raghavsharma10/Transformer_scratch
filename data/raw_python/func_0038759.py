def formulas_iter(self):
        """
        Iterates over all variable-clauses in the logical network

        Yields
        ------
        tuple[str,frozenset[caspo.core.clause.Clause]]
            The next tuple of the form (variable, set of clauses) in the logical network.
        """
        for var in it.ifilter(self.has_node, self.variables()):
            yield var, frozenset(self.predecessors(var))
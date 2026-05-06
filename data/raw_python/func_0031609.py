def to_digraph(self, *args, **kwargs):
        """
        Compute a directed graph for the chemical system.

        Returns
        -------
        digraph : `networkx.DiGraph`
            Graph nodes are reactants and/or products of chemical equations,
            while edges represent the equations themselves. Double ended edges
            are used to represent equilibria. Attributes are computed with
            `ChemicalEquation.to_series` for each equation (see examples
            below).

        Notes
        -----
        Further arguments and keywords are passed directly to
        `ChemicalEquation.to_series`.

        Examples
        --------
        >>> from pyrrole import ChemicalSystem
        >>> from pyrrole.atoms import create_data, read_cclib
        >>> data = create_data(
        ...     read_cclib("data/acetate/acetic_acid.out", "AcOH(g)"),
        ...     read_cclib("data/acetate/acetic_acid@water.out", "AcOH(aq)"))
        >>> equilibrium = ChemicalSystem("AcOH(g) <=> AcOH(aq)", data)
        >>> digraph = equilibrium.to_digraph()
        >>> sorted(digraph.nodes(data='freeenergy'))
        [('AcOH(aq)', -228.57526805), ('AcOH(g)', -228.56450866)]
        >>> digraph.number_of_nodes()
        2
        >>> digraph.number_of_edges()
        2

        """
        # TODO: make test for this
        digraph = _nx.DiGraph()
        for equation in self.equations:
            reactants, arrow, products = [value.strip() for value
                                          in _split_arrows(str(equation))]

            try:
                attr = equation.to_series("reactants", *args,
                                          **kwargs).to_dict()
            except ValueError:
                attr = dict()
            digraph.add_node(reactants, **attr)

            try:
                attr = equation.to_series("products", *args,
                                          **kwargs).to_dict()
            except ValueError:
                attr = dict()
            digraph.add_node(products, **attr)

            try:
                attr = equation.to_series(*args, **kwargs).to_dict()
            except ValueError:
                attr = dict()
            digraph.add_edge(reactants, products, **attr)
            if arrow == '<=>':
                digraph.add_edge(products, reactants, **attr)

        return digraph
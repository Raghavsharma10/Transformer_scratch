def to_dataframe(self, *args, **kwargs):
        """
        Produce a data table with records for all chemical equations.

        All possible differences for numeric attributes are computed and stored
        as columns in the returned `pandas.DataFrame` object (see examples
        below), whose rows represent chemical equations.

        In terms of behavior, this method can be seen as the `ChemicalEquation`
        counterpart of `create_data`.

        Returns
        -------
        dataframe : `pandas.DataFrame`
            Data table with records of attribute differences for every single
            `ChemicalEquation` object in the model.

        Notes
        -----
        Further arguments and keywords are passed directly to
        `ChemicalEquation.to_series`.

        Examples
        --------
        >>> from pyrrole import ChemicalSystem
        >>> from pyrrole.atoms import create_data, read_cclib
        >>> data = create_data(
        ...     read_cclib("data/acetate/acetic_acid.out",
        ...                "AcOH(g)"),
        ...     read_cclib("data/acetate/acetic_acid@water.out",
        ...                "AcOH(aq)"))
        >>> data = data[["enthalpy", "entropy", "freeenergy"]]
        >>> equilibrium = ChemicalSystem("AcOH(g) <=> AcOH(aq)", data)
        >>> equilibrium.to_dataframe()  # doctest: +NORMALIZE_WHITESPACE
                              enthalpy   entropy  freeenergy
        chemical_equation
        AcOH(g) <=> AcOH(aq) -0.010958 -0.000198   -0.010759

        """
        dataframe = _pd.DataFrame([equation.to_series(*args, **kwargs)
                                   for equation in self.equations])
        dataframe.index.name = "chemical_equation"
        return dataframe
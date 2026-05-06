def to_series(self, only=None,
                  intensive_columns=["temperature", "pressure"],
                  check_data=True):
        """
        Produce a data record for `ChemicalEquation`.

        All possible linear differences for all numeric attributes are computed
        and stored in the returned `pandas.Series` object (see examples below).
        This allows for easy application and manipulation of
        `Hess's law <https://en.wikipedia.org/wiki/Hess%27s_law>`_ to chemical
        equations (see examples below).

        Parameters
        ----------
        only : ``"reactants"``, ``"products"``, optional
            Instead of the standard behaviour (difference of sums), sum numeric
            attributes of either reactants or products only. If given, absolute
            coefficients are used.
        intensive_columns : iterable of `str`, optional
            A set of column names representing intensive properties (e.g. bulk
            properties) whose values are not summable. Those must be constant
            throughout the chemical equation.
        check_data : `bool`, optional
            Whether to check data object for inconsistencies.

        Returns
        -------
        series : `pandas.Series`
            Data record of attribute differences, whose name is the canonical
            string representation of the `ChemicalEquation` or, if `only` is
            given, a string representing either reactants or products (see
            examples below).

        Raises
        ------
        ValueError
            Raised if `self.data` wasn't defined (e.g. is `None`), if `only`
            is something other than ``"reactants"`` or ``"products"``, or if
            two or more distinct values for an intensive property have been
            found.

        Examples
        --------
        >>> from pyrrole import ChemicalEquation
        >>> from pyrrole.atoms import create_data, read_cclib
        >>> data = create_data(
        ...     read_cclib("data/acetate/acetic_acid.out",
        ...                "AcOH(g)"),
        ...     read_cclib("data/acetate/acetic_acid@water.out",
        ...                "AcOH(aq)"))
        >>> equilibrium = ChemicalEquation("AcOH(g) <=> AcOH(aq)",
        ...                                data)
        >>> equilibrium.to_series()
        charge           0.000000
        enthalpy        -0.010958
        entropy         -0.000198
        freeenergy      -0.010759
        mult             0.000000
        natom            0.000000
        nbasis           0.000000
        nmo              0.000000
        pressure         1.000000
        temperature    298.150000
        Name: AcOH(g) <=> AcOH(aq), dtype: float64

        Sums of either reactants or products can be computed:

        >>> equilibrium.to_series("reactants")
        charge           0.000000
        enthalpy      -228.533374
        entropy          0.031135
        freeenergy    -228.564509
        mult             1.000000
        natom            8.000000
        nbasis          68.000000
        nmo             68.000000
        pressure         1.000000
        temperature    298.150000
        Name: AcOH(g), dtype: float64

        """
        if self.data is None:
            # TODO: should an empty Series be returned?
            raise ValueError("data not defined")

        # TODO: find a way to keep categorical columns. Keep if they match?
        columns = self.data.select_dtypes('number').columns
        intensive_columns = [column for column in columns
                             if column in intensive_columns]
        extensive_columns = [column for column in columns
                             if column not in intensive_columns]
        columns = extensive_columns + intensive_columns

        if only is None:
            species = self.species
        elif only == "reactants":
            species = sorted(self.reactants)
        elif only == "products":
            species = sorted(self.products)
        else:
            raise ValueError("only must be either 'reactants' or 'products' "
                             "('{}' given)".format(only))

        if check_data:
            _check_data(self.data.loc[species])

        if all([s in self.data.index for s in species]):
            series = (self.data.loc[species, extensive_columns]
                      .mul(self.coefficient, axis="index").sum("index"))
            for column in intensive_columns:
                vals = self.data[column].unique()
                if len(vals) > 1:
                    raise ValueError("different values for {}: "
                                     "{}".format(column, vals))
                series[column] = vals[0]
        else:
            series = _pd.Series(_np.nan, index=columns)

        if only is None:
            name = self.__str__()
        else:
            coefficients = self.coefficient[species]
            name = _get_chemical_equation_piece(species, coefficients)
            if only == "reactants":
                series[extensive_columns] = -series[extensive_columns]

        # Avoid negative zero
        # (see https://stackoverflow.com/a/11010791/4039050)
        series = series + 0.

        return series.rename(name)
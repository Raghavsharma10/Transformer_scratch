def from_dataframe(cls, df, inhibitors=None):
        """
        Creates a list of clampings from a `pandas.DataFrame`_ object instance.
        Column names are expected to be of the form `TR:species_name`

        Parameters
        ----------
        df : `pandas.DataFrame`_
            Columns and rows correspond to species names and individual clampings, respectively.

        inhibitors : Optional[list[str]]
            If given, species names ending with `i` and found in the list (without the `i`)
            will be interpreted as inhibitors. That is, if they are set to 1, the corresponding species is inhibited
            and therefore its negatively clamped. Apart from that, all 1s (resp. 0s) are interpreted as positively
            (resp. negatively) clamped.

            Otherwise (if inhibitors=[]), all 1s (resp. -1s) are interpreted as positively (resp. negatively) clamped.


        Returns
        -------
        caspo.core.ClampingList
            Created object instance


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
        """
        inhibitors = inhibitors or []
        clampings = []
        ni = len(inhibitors)
        for _, row in df.iterrows():
            if ni > 0:
                literals = []
                for v, s in row.iteritems():
                    if v.endswith('i') and v[3:-1] in inhibitors:
                        if s == 1:
                            literals.append(Literal(v[3:-1], -1))
                    else:
                        literals.append(Literal(v[3:], 1 if s == 1 else -1))
                clampings.append(Clamping(literals))
            else:

                clampings.append(Clamping([Literal(v[3:], s) for v, s in row[row != 0].iteritems()]))

        return cls(clampings)
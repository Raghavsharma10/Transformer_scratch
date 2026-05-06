def from_csv(cls, filename, inhibitors=None):
        """
        Creates a list of clampings from a CSV file. Column names are expected to be of the form `TR:species_name`

        Parameters
        ----------
        filename : str
            Absolute path to a CSV file to be loaded with `pandas.read_csv`_. The resulting DataFrame is passed to :func:`from_dataframe`.

        inhibitors : Optional[list[str]]
            If given, species names ending with `i` and found in the list (without the `i`)
            will be interpreted as inhibitors. That is, if they are set to 1, the corresponding species is inhibited
            and therefore its negatively clamped. Apart from that, all 1s (resp. 0s) are interpreted as positively
            (resp. negatively) clamped.

            Otherwise (if inhibitors=[]), all 1s (resp. -1s) are interpreted as positively (resp. negatively) clamped.


        Returns
        -------
        caspo.core.clamping.ClampingList
            Created object instance


        .. _pandas.read_csv: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html#pandas.read_csv
        """
        df = pd.read_csv(filename)
        return cls.from_dataframe(df, inhibitors)
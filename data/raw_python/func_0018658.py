def append(self, other, inplace=False, **kwargs):
        """
        Append any input which can be converted to MAGICCData to self.

        Parameters
        ----------
        other : MAGICCData, pd.DataFrame, pd.Series, str
            Source of data to append.

        inplace : bool
            If True, append ``other`` inplace, otherwise return a new ``MAGICCData``
            instance.

        **kwargs
            Passed to ``MAGICCData`` constructor (only used if ``MAGICCData`` is not a
            ``MAGICCData`` instance).
        """
        if not isinstance(other, MAGICCData):
            other = MAGICCData(other, **kwargs)

        if inplace:
            super().append(other, inplace=inplace)
            self.metadata.update(other.metadata)
        else:
            res = super().append(other, inplace=inplace)
            res.metadata = deepcopy(self.metadata)
            res.metadata.update(other.metadata)

            return res
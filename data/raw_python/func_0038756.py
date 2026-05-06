def predictions(self, clampings, readouts, stimuli=None, inhibitors=None, nclampings=-1):
        """
        Computes network predictions for the given iterable of clampings

        Parameters
        ----------
        clampings : iterable
            Iterable over clampings

        readouts : list[str]
            List of readouts names

        stimuli : Optional[list[str]]
            List of stimuli names

        inhibitors : Optional[list[str]]
            List of inhibitors names

        nclampings : Optional[int]
            If greater than zero, it must be the number of clampings in the iterable. Otherwise,
            clampings must implement the special method :func:`__len__`


        Returns
        -------
        `pandas.DataFrame`_
            DataFrame with network predictions for each clamping. If stimuli and inhibitors are given,
            columns are included describing each clamping. Otherwise, columns correspond to readouts only.


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
        """
        stimuli, inhibitors = stimuli or [], inhibitors or []
        cues = stimuli + inhibitors
        nc = len(cues)
        ns = len(stimuli)
        predictions = np.zeros((nclampings if nclampings > 0 else len(clampings), nc+len(readouts)), dtype=np.int8)
        for i, clamping in enumerate(clampings):
            if nc > 0:
                arr = clamping.to_array(cues)
                arr[np.where(arr[:ns] == -1)[0]] = 0
                arr[ns + np.where(arr[ns:] == -1)[0]] = 1
                predictions[i, :nc] = arr

            fixpoint = self.fixpoint(clamping)
            for j, readout in enumerate(readouts):
                predictions[i, nc+j] = fixpoint.get(readout, 0)

        return pd.DataFrame(predictions, columns=np.concatenate([stimuli, [i+'i' for i in inhibitors], readouts]))
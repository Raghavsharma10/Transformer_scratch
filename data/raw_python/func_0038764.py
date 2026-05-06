def to_dataframe(self, stimuli=None, inhibitors=None, prepend=""):
        """
        Converts the list of clampigns to a `pandas.DataFrame`_ object instance

        Parameters
        ----------
        stimuli : Optional[list[str]]
            List of stimuli names. If given, stimuli are converted to {0,1} instead of {-1,1}.

        inhibitors : Optional[list[str]]
            List of inhibitors names. If given, inhibitors are renamed and converted to {0,1} instead of {-1,1}.

        prepend : str
            Columns are renamed using the given string at the beginning

        Returns
        -------
        `pandas.DataFrame`_
            DataFrame representation of the list of clampings


        .. _pandas.DataFrame: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
        """
        stimuli, inhibitors = stimuli or [], inhibitors or []
        cues = stimuli + inhibitors
        nc = len(cues)
        ns = len(stimuli)

        variables = cues or np.array(list(set((v for (v, s) in it.chain.from_iterable(self)))))

        matrix = np.array([])
        for clamping in self:
            arr = clamping.to_array(variables)
            if nc > 0:
                arr[np.where(arr[:ns] == -1)[0]] = 0
                arr[ns + np.where(arr[ns:] == -1)[0]] = 1

            if len(matrix):
                matrix = np.append(matrix, [arr], axis=0)
            else:
                matrix = np.array([arr])

        return pd.DataFrame(matrix, columns=[prepend + "%s" % c for c in (stimuli + [i+'i' for i in inhibitors] if nc > 0 else variables)])
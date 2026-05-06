def _convert_data_block_and_headers_to_df(self, stream):
        """
        stream : Streamlike object
            A Streamlike object (nominally StringIO) containing the data to be
            extracted

        ch : dict
            Column headers to use for the output pd.DataFrame

        Returns
        -------
        :obj:`pd.DataFrame`
            Dataframe with processed datablock
        """
        df = pd.read_csv(
            stream,
            skip_blank_lines=True,
            delim_whitespace=True,
            header=None,
            index_col=0,
        )

        if isinstance(df.index, pd.core.indexes.numeric.Float64Index):
            df.index = df.index.to_series().round(3)

        # reset the columns to be 0..n instead of starting at 1
        df.columns = list(range(len(df.columns)))

        return df
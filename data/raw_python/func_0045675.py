def pandas_dataframe(self, start, stop, ncol, **kwargs):
        """
        Returns the result of tab-separated pandas.read_csv on
        a subset of the file.

        Args:
            start (int): line number where structured data starts
            stop (int): line number where structured data stops
            ncol (int or list): the number of columns in the structured
                data or a list of that length with column names

        Returns:
            pd.DataFrame: structured data
        """
        try:
            int(start)
            int(stop)
        except TypeError:
            print('start and stop must be ints')
        try:
            ncol = int(ncol)
            return pd.read_csv(six.StringIO('\n'.join(self[start:stop])), delim_whitespace=True, names=range(ncol), **kwargs)
        except TypeError:
            try:
                ncol = list(ncol)
                return pd.read_csv(six.StringIO('\n'.join(self[start:stop])), delim_whitespace=True, names=ncol, **kwargs)
            except TypeError:
                print('Cannot pandas_dataframe if ncol is {}, must be int or list'.format(type(ncol)))
def unpack(self, name):
        """
        Unpacks a data set to a Pandas DataFrame

        Parameters
        ----------
        name : str
            call `.list` to see all availble datasets

        Returns
        -------
        pd.DataFrame
        """
        path = self.list[name]
        df = pd.read_pickle(path, compression='gzip')
        return df
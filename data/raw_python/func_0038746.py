def to_csv(self, filename, networks=False, dataset=None, size=False, n_jobs=-1):
        """
        Writes the list of logical networks to a CSV file

        Parameters
        ----------
        filename : str
            Absolute path where to write the CSV file

        networks : boolean
            If True, a column with number of networks having the same behavior is included in the file

        dataset: Optional[:class:`caspo.core.dataset.Dataset`]
            If not None, a column with the MSE with respect to the given dataset is included

        size: boolean
            If True, a column with the size of each logical network is included

        n_jobs : int
            Number of jobs to run in parallel. Default to -1 (all cores available)

        """
        self.to_dataframe(networks, dataset, size, n_jobs).to_csv(filename, index=False)
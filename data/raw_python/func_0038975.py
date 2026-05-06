def random_subset(self, perc_in_class=0.5):
        """
        Returns a random sub-dataset (of specified size by percentage) within each class.

        Parameters
        ----------
        perc_in_class : float
            Fraction of samples to be taken from each class.

        Returns
        -------
        subdataset : MLDataset
            random sub-dataset of specified size.

        """

        subsets = self.random_subset_ids(perc_in_class)
        if len(subsets) > 0:
            return self.get_subset(subsets)
        else:
            warnings.warn('Zero samples were selected. Returning an empty dataset!')
            return MLDataset()
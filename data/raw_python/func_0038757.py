def mse(self, dataset):
        """
        Returns the Mean Squared Error with respect to the given :class:`caspo.core.dataset.Dataset` object

        Parameters
        ----------
        dataset : :class:`caspo.core.dataset.Dataset`
            Dataset to compute MSE

        Returns
        -------
        float
            Computed mean squared error
        """
        clampings = dataset.clampings
        readouts = dataset.readouts.columns
        observations = dataset.readouts.values
        pos = ~np.isnan(observations)

        return mean_squared_error(observations, (self.predictions(clampings, readouts).values)[pos])
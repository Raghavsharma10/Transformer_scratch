def weighted_mse(self, dataset, n_jobs=-1):
        """
        Returns the weighted MSE over all logical networks with respect to the given :class:`caspo.core.dataset.Dataset` object instance.
        For each logical network the weight corresponds to the number of networks having the same behavior.

        Parameters
        ----------
        dataset: :class:`caspo.core.dataset.Dataset`
            Dataset to compute MSE

        n_jobs : int
            Number of jobs to run in parallel. Default to -1 (all cores available)

        Returns
        -------
        float
            Weighted MSE
        """
        predictions = np.zeros((len(self), len(dataset.clampings), len(dataset.setup.readouts)))
        predictions[:, :, :] = Parallel(n_jobs=n_jobs)(delayed(__parallel_predictions__)(n, dataset.clampings, dataset.setup.readouts) for n in self)
        for i, _ in enumerate(self):
            predictions[i, :, :] *= self.__networks[i]

        readouts = dataset.readouts.values
        pos = ~np.isnan(readouts)

        return mean_squared_error(readouts[pos], (np.sum(predictions, axis=0) / np.sum(self.__networks))[pos])
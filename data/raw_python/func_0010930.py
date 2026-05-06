def _step(self, dataset):
        '''Advance the state of the optimizer by one step.

        Parameters
        ----------
        dataset : :class:`Dataset <downhill.dataset.Dataset>`
            A dataset for optimizing the model.

        Returns
        -------
        train_monitors : dict
            A dictionary mapping monitor names to values.
        '''
        if dataset is None:
            values = [self.f_step()]
        else:
            values = [self.f_step(*x) for x in dataset]
        return collections.OrderedDict(
            zip(self._monitor_names, np.mean(values, axis=0)))
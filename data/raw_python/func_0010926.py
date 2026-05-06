def evaluate(self, dataset):
        '''Evaluate the current model parameters on a dataset.

        Parameters
        ----------
        dataset : :class:`Dataset <downhill.dataset.Dataset>`
            A set of data to use for evaluating the model.

        Returns
        -------
        monitors : OrderedDict
            A dictionary mapping monitor names to values. Monitors are
            quantities of interest during optimization---for example, loss
            function, accuracy, or whatever the optimization task requires.
        '''
        if dataset is None:
            values = [self.f_eval()]
        else:
            values = [self.f_eval(*x) for x in dataset]
        monitors = zip(self._monitor_names, np.mean(values, axis=0))
        return collections.OrderedDict(monitors)
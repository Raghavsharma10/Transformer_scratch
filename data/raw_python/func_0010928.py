def iterate(self, train=None, valid=None, max_updates=None, **kwargs):
        r'''Optimize a loss iteratively using a training and validation dataset.

        This method yields a series of monitor values to the caller. After every
        optimization epoch, a pair of monitor dictionaries is generated: one
        evaluated on the training dataset during the epoch, and another
        evaluated on the validation dataset at the most recent validation epoch.

        The validation monitors might not be updated during every optimization
        iteration; in this case, the most recent validation monitors will be
        yielded along with the training monitors.

        Additional keyword arguments supplied here will set the global
        optimizer attributes.

        Parameters
        ----------
        train : sequence or :class:`Dataset <downhill.dataset.Dataset>`
            A set of training data for computing updates to model parameters.
        valid : sequence or :class:`Dataset <downhill.dataset.Dataset>`
            A set of validation data for computing monitor values and
            determining when the loss has stopped improving. Defaults to the
            training data.
        max_updates : int, optional
            If specified, halt optimization after this many gradient updates
            have been processed. If not provided, uses early stopping to decide
            when to halt.

        Yields
        ------
        train_monitors : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        valid_monitors : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        self._compile(**kwargs)

        if valid is None:
            valid = train
        iteration = 0
        training = validation = None
        while max_updates is None or iteration < max_updates:
            if not iteration % self.validate_every:
                try:
                    validation = self.evaluate(valid)
                except KeyboardInterrupt:
                    util.log('interrupted!')
                    break
                if self._test_patience(validation):
                    util.log('patience elapsed!')
                    break
            try:
                training = self._step(train)
            except KeyboardInterrupt:
                util.log('interrupted!')
                break
            iteration += 1
            self._log(training, iteration)
            yield training, validation
        self.set_params('best')
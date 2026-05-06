def minimize(self, *args, **kwargs):
        '''Optimize our loss exhaustively.

        This method is a thin wrapper over the :func:`iterate` method. It simply
        exhausts the iterative optimization process and returns the final
        monitor values.

        Returns
        -------
        train_monitors : dict
            A dictionary mapping monitor names to values, evaluated on the
            training dataset.
        valid_monitors : dict
            A dictionary containing monitor values evaluated on the validation
            dataset.
        '''
        monitors = None
        for monitors in self.iterate(*args, **kwargs):
            pass
        return monitors
def iterate(self, shuffle=True):
        '''Iterate over batches in the dataset.

        This method generates ``iteration_size`` batches from the dataset and
        then returns.

        Parameters
        ----------
        shuffle : bool, optional
            Shuffle the batches in this dataset if the iteration reaches the end
            of the batch list. Defaults to True.

        Yields
        ------
        batches : data batches
            A sequence of batches---often from a training, validation, or test
            dataset.
        '''
        for _ in range(self.iteration_size):
            if self._callable is not None:
                yield self._callable()
            else:
                yield self._next_batch(shuffle)
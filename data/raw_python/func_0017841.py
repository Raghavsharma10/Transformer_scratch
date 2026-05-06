def batch_map_concat(self, func, batch_size, progress_iter_func=None,
                         n_batches=None, prepend_args=None, **kwargs):
        """A batch oriented implementation of `map`.
        Applies a function to all the samples in this data source by breaking
        the data into mini-batches and applying the function to each
        mini-batch.
        Returns the per-sample results.

        This method is a wrapper around the :func:`batch_map` function;
        please see its documentation for more information and examples.

        The function `func` should return the result for each sample in the
        mini-batch as an array. To return multiple results (e.g. loss and
        errors) return a tuple of arrays (e.g. `(loss_array, error_array)`)

        Parameters
        ----------
        func: callable `func(*batch) -> results`
            The function to call on each mini-batch. Note that the results
            must be `None`, a tuple or a NumPy array
        batch_size: int
            The mini-batch size
        progress_iter_func: [optional] callable
            `progress_iter_func(iterator, total=total, leave=leave)`
            A `tqdm` style function that will be passed the iterator that
            generates training batches along with the total number of batches
            and `False` for the `leave` parameter. By passing either
            `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
            the training loop display a progress bar.
        n_batches: [optional] integer that specifies the number of mini-batches
            to process before returning
        prepend_args: [optional] tuple
            Arguments to prepend to the arguments passed to `func`

        Returns
        -------
        tuple
            The per-sample sum of the results of the function `func` e.g.
            `(batch_A, batch_B, ...)`
            Returns an empty tuple if there were 0 samples in the data set.

        Examples
        --------
        Define a function to apply to samples:
        >>> def sqr_sum(x):
        ...     return (x ** 2).sum(axis=1)

        Construct data to process and create a data source:
        >>> X = np.random.normal(size=(7, 10))
        >>> ds = ArrayDataSource([X])

        Apply the function defined above:
        >>> X_sqr_sum = ds.batch_map_concat(sqr_sum, batch_size=5)
        >>> assert (X_sqr_sum[0] == (X ** 2).sum(axis=1)).all()
        """
        if n_batches is None:
            n = self.num_samples(**kwargs)
            if n == np.inf:
                raise ValueError('Data set has infinite size or sampler will '
                                 'generate infinite samples but no n_batches '
                                 'limit specified')
            elif n is not None:
                n_batches = sampling.num_batches(n, batch_size)
        batch_iter = self.batch_iterator(batch_size, **kwargs)
        return batch_map_concat(func, batch_iter, progress_iter_func,
                                n_batches, prepend_args)
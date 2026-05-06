def batch_map_concat(func, batch_iter, progress_iter_func=None,
                     n_batches=None, prepend_args=None):
    """
    Apply a function to all the samples that are accessed as mini-batches
    obtained from an iterator.
    Returns the per-sample results.

    The function `func` should return the result for each sample in the
    mini-batch as an array. To return multiple results (e.g. loss and errors)
    return a tuple of arrays (e.g. `(loss_array, error_array)`)

    `batch_iter` must be an iterator that generates mini-batches that
    contain samples

    Parameters
    ----------
    func: callable `func(*batch) -> results`
        The function to call on each mini-batch. Note that the results
        must be `None`, a tuple or a NumPy array
    batch_iter: data set iterator
        Iterator that generates mini-batches of data
    progress_iter_func: [optional] callable
        `progress_iter_func(iterator, total=total, leave=leave)`
        A `tqdm` style function that will be passed the iterator that
        generates training batches along with the total number of batches
        and `False` for the `leave` parameter. By passing either
        `tqdm.tqdm` or `tqdm.tqdm_notebook` as this argument you can have
        the training loop display a progress bar.
    n_batches: [optional] integer
        Process at most this number of batches before returning.
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
    In these examples we will demonstrate the use of `batch_map` to apply
    a function (e.g. a Theano function that runs on the GPU) to samples
    in a data set. We construct an iterator that generates mini-batches from
    the data set and pass it to `batch_map` along with the function that
    we wish to apply. The function will receive the batches and process them.

    Define a function to apply to samples:
    >>> def sqr_sum(x):
    ...     # Ensure that we receive batches of the expected size:
    ...     assert len(x) in {5, 2}
    ...     return (x ** 2).sum(axis=1)

    Construct data to process and create a data source:
    >>> X = np.random.normal(size=(7, 10))
    >>> ds = ArrayDataSource([X])

    Apply the function defined above:
    >>> batch_iter = ds.batch_iterator(batch_size=5)
    >>> X_sqr_sum = batch_map_concat(sqr_sum, batch_iter)
    >>> assert np.allclose(X_sqr_sum[0], (X ** 2).sum(axis=1))

    There are also cases where we wish to limit the number of batches that
    will be processed:
    - when the iterator generates an infinite number of samples
    - when the data set is huge and we wish to show results as we go
    Use the `n_batches` argument to limit the number of batches to process:
    >>> X_large = np.random.normal(size=(100, 10))
    >>> ds_large = ArrayDataSource([X_large])
    >>> iter_large = ds_large.batch_iterator(batch_size=5)
    >>> for i in range(10):
    ...     partial_result = batch_map_concat(sqr_sum, iter_large, n_batches=2)
    ...     # Should have 10 samples per partial result
    ...     assert len(partial_result[0]) == 10
    ...     j = i * 10
    ...     assert np.allclose(partial_result[0],
    ...                        (X_large[j:j + 10]**2).sum(axis=1))
    """
    # Accumulator for results and number of samples
    results = []

    # If `progress_iter_func` is not `None`, apply it
    if progress_iter_func is not None:
        batch_iter = progress_iter_func(batch_iter, total=n_batches,
                                        leave=False)

    # Apply `func` to each batch
    n_processed = 0
    for batch in batch_iter:
        # Apply on batch and check the type of the results
        if prepend_args is not None:
            batch_results = func(*(prepend_args + tuple(batch)))
        else:
            batch_results = func(*batch)
        if batch_results is None:
            pass
        elif isinstance(batch_results, np.ndarray):
            batch_results = (batch_results,)
        elif isinstance(batch_results, tuple):
            pass
        else:
            raise TypeError(
                    'Batch function should return a tuple of results, a '
                    'single result as a NumPy array, or None, '
                    'not {}'.format(type(batch_results)))

        # Accumulate training results
        if batch_results is not None:
            results.append(batch_results)

        n_processed += 1
        if n_batches is not None and n_processed >= n_batches:
            break

    # Concatenate result arrays
    if len(results) > 0:
        results = zip(*results)
        results = tuple([np.concatenate(list(r), axis=0) for r in results])
        return results
    else:
        return None
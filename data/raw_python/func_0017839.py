def batch_map_mean(func, batch_iter, progress_iter_func=None, sum_axis=None,
                   n_batches=None, prepend_args=None):
    """
    Apply a function to all the samples that are accessed as mini-batches
    obtained from an iterator.
    Returns the across-samples mean of the results returned by `func`

    The `sum_axis` arguments tells `mean_batch_map` how to process the
    results of `func` before accumulating them:
    - If `sum_axis` is `None`, `func` should return the
    across-samples SUM of the  results of operating on the mini-batch the
    sum of the values for the samples, e.g. for loss and error it should
    return `(sum([loss0, loss1, ... lossN]), sum([err0, err1, ... errN]))`
    - Otherwise, `sum_axis` should specify the axis or axes over which
    the the batch results should be summed, e.g. if `func` returns a
    per-sample loss and error in two arrays
    `[[loss0, loss1, ... lossN], [err0, err1, ... errN]`, give `sum_axis`
    a value of `0` to sum over axis 0 to get the per-batch loss and error.
    These results will be accumulated and divided by the number of samples
    at the end to get the mean.

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
    sum_axis: (default=`None`) int, tuple of ints or None
        If an integer or a tuple of integers, the results returned by `func`
        will be summed across this axis / these axes before being accumulated;
        e.g. if `func` returns an array of per-sample losses, with axis 0
        being the sample dimension, passing a value of `0` as `sum_axis`
        will cause these results to be summed along axis 0 to get the
        per-batch sum before accumulating the losses. The total summed loss
        will be divided by the number of samples at the end in order to
        compute the mean loss.
    n_batches: [optional] integer that specifies the number of mini-batches
        to process before returning
    prepend_args: [optional] tuple
        Arguments to prepend to the arguments passed to `func`

    Returns
    -------
    tuple
        The sum of the results of the function `fn` divided by the number of
        samples processed, e.g.
        `(sum(outA_per_batch) / n_samples,
          sum(outB_per_batch) / n_samples,
          ...)`

    Examples
    --------
    The following examples will demonstrate the use of `mean_batch_map`
    to compute binary cross entropy loss over a data set.
    A few variants will be demonstrated:
    - the default behaviour in which the function being applied should
      return the sum over the batch sample axis
    - having the function return per sample results and maving
      `mean_batch_map` perform the sum operation. This is easier to
      understand but less efficient as a Theano function would have to
      move more data back from the GPU.
    - limiting the number of batches that will be processed in order to get
      partial results when dealing with a large data set

    Define a function to compute the per-sample binary cross entropy
    loss:
    >>> def binary_crossentropy_loss(pred, target):
    ...     e = -target * np.log(pred) - (1 - target) * np.log(1 - pred)
    ...     return e.mean(axis=1)

    Now define a function that computes the *SUM* of the binary cross
    entropy losses over the sample axis (axis 0), as the default
    behaviour of `mean_batch_map` will sum them up and divide by the
    number of samples at the end:
    >>> def binary_crossentropy_loss_sum(pred, target):
    ...     return binary_crossentropy_loss(pred, target).sum()

    Construct prediction and target data
    >>> pred = np.random.uniform(0.1, 0.9, size=(7, 10))
    >>> tgt = np.random.uniform(0.1, 0.9, size=(7, 10))
    >>> ds = ArrayDataSource([pred, tgt])

    Apply the loss sum function defined above:
    >>> batch_iter = ds.batch_iterator(batch_size=5)
    >>> loss = batch_map_mean(binary_crossentropy_loss_sum, batch_iter)
    >>> assert np.allclose(
    ...     loss, binary_crossentropy_loss(pred, tgt).mean())

    Have `mean_batch_map` sum over axis 0:
    >>> batch_iter = ds.batch_iterator(batch_size=5)
    >>> loss = batch_map_mean(binary_crossentropy_loss, batch_iter,
    ...                       sum_axis=0)
    >>> assert np.allclose(
    ...     loss, binary_crossentropy_loss(pred, tgt).mean())

    Construct a large data set and use `batch
    >>> pred_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    >>> tgt_large = np.random.uniform(0.1, 0.9, size=(100, 10))
    >>> ds_large = ArrayDataSource([pred_large, tgt_large])
    >>> iter_large = ds_large.batch_iterator(batch_size=5)
    >>> for i in range(10):
    ...     partial_loss = batch_map_mean(binary_crossentropy_loss_sum,
    ...                                   iter_large, n_batches=2)
    ...     j = i * 10
    ...     assert np.allclose(
    ...         partial_loss, binary_crossentropy_loss(
    ...             pred_large[j:j + 10], tgt_large[j:j + 10]).mean())
    """
    # Accumulator for results and number of samples
    results_accum = None
    n_samples_accum = 0

    # If `progress_iter_func` is not `None`, apply it
    if progress_iter_func is not None:
        batch_iter = progress_iter_func(batch_iter, total=n_batches,
                                        leave=False)

    # Train on each batch
    n_processed = 0
    for batch in batch_iter:
        # Get number of samples in batch; can vary
        batch_n = _length_of_batch(batch)

        # Apply on batch and check the type of the results
        if prepend_args is not None:
            batch_results = func(*(prepend_args + tuple(batch)))
        else:
            batch_results = func(*batch)
        if batch_results is None:
            pass
        elif isinstance(batch_results, (np.ndarray, float)):
            batch_results = (batch_results,)
        elif isinstance(batch_results, tuple):
            pass
        else:
            raise TypeError(
                    'Batch function should return a tuple of results, a '
                    'single result as a NumPy array or float, or None, '
                    'not {}'.format(type(batch_results)))

        # Accumulate results and number of samples
        if results_accum is None:
            # Initialise the accumulator to the batch results if `func`
            # returns summed results or if it returned None;
            # don't attempt to iterate over None and sum each item
            if batch_results is None:
                pass
            elif sum_axis is None:
                results_accum = list(batch_results)
            else:
                results_accum = [br.sum(axis=sum_axis) for br in batch_results]
        else:
            if batch_results is not None:
                for i in range(len(results_accum)):
                    br = batch_results[i]
                    if sum_axis is not None:
                        br = br.sum(axis=sum_axis)
                    results_accum[i] += br
        n_samples_accum += batch_n

        n_processed += 1
        if n_batches is not None and n_processed >= n_batches:
            break

    # Divide by the number of training examples used to compute mean
    if results_accum is not None:
        results_accum = tuple([np.array(r).astype(float) / n_samples_accum
                               for r in results_accum])

    return results_accum
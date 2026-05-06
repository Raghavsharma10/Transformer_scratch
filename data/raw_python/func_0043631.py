def calculate_num_slices_per_axis(num_slices_per_axis, num_slices, max_slices_per_axis=None):
    """
    Returns a :obj:`numpy.ndarray` (:samp:`return_array` say) where non-positive elements of
    the :samp:`{num_slices_per_axis}` sequence have been replaced with
    positive integer values such that :samp:`numpy.product(return_array) == num_slices`
    and::

       numpy.all(
           return_array[numpy.where(num_slices_per_axis <= 0)]
           <=
           max_slices_per_axis[numpy.where(num_slices_per_axis <= 0)]
       ) is True


    :type num_slices_per_axis: sequence of :obj:`int`
    :param num_slices_per_axis: Constraint for per-axis sub-divisions.
       Non-positive elements indicate values to be replaced in the
       returned array. Positive values are identical to the corresponding
       element in the returned array.
    :type num_slices: integer
    :param num_slices: Indicates the number of slices (rectangular sub-arrays)
       formed by performing sub-divisions per axis. The returned array :samp:`return_array`
       has elements assigned such that :samp:`numpy.product(return_array) == {num_slices}`.
    :type max_slices_per_axis: sequence of :obj:`int` (or :samp:`None`)
    :param max_slices_per_axis: Constraint specifying maximum number of per-axis sub-divisions.
       If :samp:`None` defaults to :samp:`numpy.array([numpy.inf,]*len({num_slices_per_axis}))`.
    :rtype: :obj:`numpy.ndarray`
    :return: An array :samp:`return_array`
       such that :samp:`numpy.product(return_array) == num_slices`.


    Examples::

       >>> from array_split.split import calculate_num_slices_per_axis
       >>>
       >>> calculate_num_slices_per_axis([0, 0, 0], 16)
       array([4, 2, 2])
       >>> calculate_num_slices_per_axis([1, 0, 0], 16)
       array([1, 4, 4])
       >>> calculate_num_slices_per_axis([1, 0, 0], 16, [2, 2, 16])
       array([1, 2, 8])


    """
    logger = _logging.getLogger(__name__)

    ret_array = _np.array(num_slices_per_axis, copy=True)
    if max_slices_per_axis is None:
        max_slices_per_axis = _np.array([_np.inf, ] * len(num_slices_per_axis))

    max_slices_per_axis = _np.array(max_slices_per_axis)

    if _np.any(max_slices_per_axis <= 0):
        raise ValueError("Got non-positive value in max_slices_per_axis=%s" % max_slices_per_axis)

    while _np.any(ret_array <= 0):
        prd = _np.product(ret_array[_np.where(ret_array > 0)])  # returns 1 for zero-length array
        if (num_slices < prd) or ((num_slices % prd) > 0):
            raise ValueError(
                (
                    "Unable to construct grid of num_slices=%s elements from "
                    +
                    "num_slices_per_axis=%s (with max_slices_per_axis=%s)"
                )
                %
                (num_slices, num_slices_per_axis, max_slices_per_axis)
            )
        ridx = _np.where(ret_array <= 0)
        f = shape_factors(num_slices // prd, ridx[0].shape[0])[::-1]
        if _np.all(f < max_slices_per_axis[ridx]):
            ret_array[ridx] = f
        else:
            for i in range(ridx[0].shape[0]):
                if f[i] >= max_slices_per_axis[ridx[0][i]]:
                    ret_array[ridx[0][i]] = max_slices_per_axis[ridx[0][i]]
                    prd = _np.product(ret_array[_np.where(ret_array > 0)])
                    while (num_slices % prd) > 0:
                        ret_array[ridx[0][i]] -= 1
                        prd = _np.product(ret_array[_np.where(ret_array > 0)])
        logger.debug(
            "ridx=%s, f=%s, ret_array=%s, max_slices_per_axis=%s",
            ridx, f, ret_array, max_slices_per_axis
        )
    return ret_array
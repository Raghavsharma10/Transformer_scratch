def reduce_sorted_to_intersect(ar1, ar2):
    """
    Takes two sorted arrays and return the intersection ar1 in ar2, ar2 in ar1.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
         Input array.

    Returns
    -------
    ar1, ar1 : ndarray, ndarray
        The intersection values.

    """
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # get min max values of the arrays
    ar1_biggest_value = ar1[-1]
    ar1_smallest_value = ar1[0]
    ar2_biggest_value = ar2[-1]
    ar2_smallest_value = ar2[0]

    if ar1_biggest_value < ar2_smallest_value or ar1_smallest_value > ar2_biggest_value:  # special case, no intersection at all
        return ar1[0:0], ar2[0:0]

    # get min/max indices with values that are also in the other array
    min_index_ar1 = np.argmin(ar1 < ar2_smallest_value)
    max_index_ar1 = np.argmax(ar1 > ar2_biggest_value)
    min_index_ar2 = np.argmin(ar2 < ar1_smallest_value)
    max_index_ar2 = np.argmax(ar2 > ar1_biggest_value)

    if min_index_ar1 < 0:
        min_index_ar1 = 0
    if min_index_ar2 < 0:
        min_index_ar2 = 0
    if max_index_ar1 == 0 or max_index_ar1 > ar1.shape[0]:
        max_index_ar1 = ar1.shape[0]
    if max_index_ar2 == 0 or max_index_ar2 > ar2.shape[0]:
        max_index_ar2 = ar2.shape[0]

    # reduce the data
    return ar1[min_index_ar1:max_index_ar1], ar2[min_index_ar2:max_index_ar2]
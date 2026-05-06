def calc_mean_time(timepoints, weights):
    """Return the weighted mean of the given timepoints.

    With equal given weights, the result is simply the mean of the given
    time points:

    >>> from hydpy import calc_mean_time
    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[2., 2.])
    5.0

    With different weights, the resulting mean time is shifted to the larger
    ones:

    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[1., 3.])
    6.0

    Or, in the most extreme case:

    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[0., 4.])
    7.0

    There will be some checks for input plausibility perfomed, e.g.:

    >>> calc_mean_time(timepoints=[3., 7.],
    ...                weights=[-2., 2.])
    Traceback (most recent call last):
    ...
    ValueError: While trying to calculate the weighted mean time, \
the following error occurred: For the following objects, at least \
one value is negative: weights.
    """
    timepoints = numpy.array(timepoints)
    weights = numpy.array(weights)
    validtools.test_equal_shape(timepoints=timepoints, weights=weights)
    validtools.test_non_negative(weights=weights)
    return numpy.dot(timepoints, weights)/numpy.sum(weights)
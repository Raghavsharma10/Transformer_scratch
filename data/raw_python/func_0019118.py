def calc_mean_time_deviation(timepoints, weights, mean_time=None):
    """Return the weighted deviation of the given timepoints from their mean
    time.

    With equal given weights, the is simply the standard deviation of the
    given time points:

    >>> from hydpy import calc_mean_time_deviation
    >>> calc_mean_time_deviation(timepoints=[3., 7.],
    ...                          weights=[2., 2.])
    2.0

    One can pass a precalculated or alternate mean time:

    >>> from hydpy import round_
    >>> round_(calc_mean_time_deviation(timepoints=[3., 7.],
    ...                                 weights=[2., 2.],
    ...                                 mean_time=4.))
    2.236068

    >>> round_(calc_mean_time_deviation(timepoints=[3., 7.],
    ...                                 weights=[1., 3.]))
    1.732051

    Or, in the most extreme case:

    >>> calc_mean_time_deviation(timepoints=[3., 7.],
    ...                          weights=[0., 4.])
    0.0

    There will be some checks for input plausibility perfomed, e.g.:

    >>> calc_mean_time_deviation(timepoints=[3., 7.],
    ...                          weights=[-2., 2.])
    Traceback (most recent call last):
    ...
    ValueError: While trying to calculate the weighted time deviation \
from mean time, the following error occurred: For the following objects, \
at least one value is negative: weights.
    """
    timepoints = numpy.array(timepoints)
    weights = numpy.array(weights)
    validtools.test_equal_shape(timepoints=timepoints, weights=weights)
    validtools.test_non_negative(weights=weights)
    if mean_time is None:
        mean_time = calc_mean_time(timepoints, weights)
    return (numpy.sqrt(numpy.dot(weights, (timepoints-mean_time)**2) /
                       numpy.sum(weights)))
def _build_repeat_suffix(iteration, count):
    """
    Return the suffix string to identify iteration X out of Y.

    For example, with a count of 100, this will build strings like
    "iteration_053" or "iteration_008".

    :param iteration:
        Current iteration.
    :type iteration:
        `int`
    :param count:
        Total number of iterations.
    :type count:
        `int`
    :return:
        Repeat suffix.
    :rtype:
        `unicode`
    """
    format_width = int(math.ceil(math.log(count + 1, 10)))
    new_suffix = 'iteration_{0:0{width}d}'.format(
        iteration,
        width=format_width
    )
    return new_suffix
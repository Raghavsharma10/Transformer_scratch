def vectorize(fn):
    """
    Allows a method to accept one or more values,
    but internally deal only with a single item,
    and returning a list or a single item depending
    on what is desired.
    """

    @functools.wraps(fn)
    def vectorized_method(self, values, *vargs, **kwargs):
        wrap = not isinstance(values, (list, tuple))
        should_unwrap = not kwargs.setdefault('wrap', False)
        unwrap = wrap and should_unwrap
        del kwargs['wrap']
        
        if wrap:
            values = [values]

        results = [fn(self, value, *vargs, **kwargs) for value in values]

        if unwrap:
            results = results[0]

        return results

    return vectorized_method
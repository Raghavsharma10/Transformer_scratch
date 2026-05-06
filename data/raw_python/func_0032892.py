def check_shape(meth):
    """
    Decorator for larray magic methods, to ensure that the operand has
    the same shape as the array.
    """
    @wraps(meth)
    def wrapped_meth(self, val):
        if isinstance(val, (larray, numpy.ndarray)):
            if val.shape != self._shape:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        return meth(self, val)
    return wrapped_meth
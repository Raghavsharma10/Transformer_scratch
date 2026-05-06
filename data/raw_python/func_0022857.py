def arg_to_vec4(func, self_, arg, *args, **kwargs):
    """
    Decorator for converting argument to vec4 format suitable for 4x4 matrix
    multiplication.

    [x, y]      =>  [[x, y, 0, 1]]

    [x, y, z]   =>  [[x, y, z, 1]]

    [[x1, y1],      [[x1, y1, 0, 1],
     [x2, y2],  =>   [x2, y2, 0, 1],
     [x3, y3]]       [x3, y3, 0, 1]]

    If 1D input is provided, then the return value will be flattened.
    Accepts input of any dimension, as long as shape[-1] <= 4

    Alternatively, any class may define its own transform conversion interface
    by defining a _transform_in() method that returns an array with shape
    (.., 4), and a _transform_out() method that accepts the same array shape
    and returns a new (mapped) object.

    """
    if isinstance(arg, (tuple, list, np.ndarray)):
        arg = np.array(arg)
        flatten = arg.ndim == 1
        arg = as_vec4(arg)

        ret = func(self_, arg, *args, **kwargs)
        if flatten and ret is not None:
            return ret.flatten()
        return ret
    elif hasattr(arg, '_transform_in'):
        arr = arg._transform_in()
        ret = func(self_, arr, *args, **kwargs)
        return arg._transform_out(ret)
    else:
        raise TypeError("Cannot convert argument to 4D vector: %s" % arg)
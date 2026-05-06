def keras_tuples(stream, inputs=None, outputs=None):
    """Reformat data objects as keras-compatible tuples.

    For more detail: https://keras.io/models/model/#fit

    Parameters
    ----------
    stream : iterable
        Stream of data objects.

    inputs : string or iterable of strings, None
        Keys to use for ordered input data.
        If not specified, returns `None` in its place.

    outputs : string or iterable of strings, default=None
        Keys to use for ordered output data.
        If not specified, returns `None` in its place.

    Yields
    ------
    x : np.ndarray, list of np.ndarray, or None
        If `inputs` is a string, `x` is a single np.ndarray.
        If `inputs` is an iterable of strings, `x` is a list of np.ndarrays.
        If `inputs` is a null type, `x` is None.
    y : np.ndarray, list of np.ndarray, or None
        If `outputs` is a string, `y` is a single np.ndarray.
        If `outputs` is an iterable of strings, `y` is a list of np.ndarrays.
        If `outputs` is a null type, `y` is None.

    Raises
    ------
    DataError
        If the stream contains items that are not data-like.
    """
    flatten_inputs, flatten_outputs = False, False
    if inputs and isinstance(inputs, six.string_types):
        inputs = [inputs]
        flatten_inputs = True
    if outputs and isinstance(outputs, six.string_types):
        outputs = [outputs]
        flatten_outputs = True

    inputs, outputs = (inputs or []), (outputs or [])
    if not inputs + outputs:
        raise PescadorError('At least one key must be given for '
                            '`inputs` or `outputs`')

    for data in stream:
        try:
            x = list(data[key] for key in inputs) or None
            if len(inputs) == 1 and flatten_inputs:
                x = x[0]

            y = list(data[key] for key in outputs) or None
            if len(outputs) == 1 and flatten_outputs:
                y = y[0]

            yield (x, y)
        except TypeError:
            raise DataError("Malformed data stream: {}".format(data))
def _convert_operator(op_name, attrs, identity_list=None, convert_map=None):
    """Convert from onnx operator to mxnet operator.
    The converter must specify conversions explicitly for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, FullyConnected
    attrs : dict
        Dict of operator attributes
    identity_list : list
        List of operators that don't require conversion
    convert_map : dict
        Dict of name : callable, where name is the op's name that
        require conversion to mxnet, callable are functions which
        take attrs and return (new_op_name, new_attrs)

    Returns
    -------
    (op_name, attrs)
        Converted (op_name, attrs) for mxnet.
    """
    identity_list = identity_list if identity_list else _identity_list
    convert_map = convert_map if convert_map else _convert_map
    if op_name in identity_list:
        pass
    elif op_name in convert_map:
        op_name, attrs = convert_map[op_name](attrs)
    else:
        raise NotImplementedError("Operator {} not implemented.".format(op_name))
    op = getattr(mx.sym, op_name, None)
    if not op:
        raise RuntimeError("Unable to map op_name {} to sym".format(op_name))
    return op, attrs
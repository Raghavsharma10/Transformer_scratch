def _kwargs_from_call(param_names: List[str], kwdefaults: Dict[str, Any], args: Tuple[Any, ...],
                      kwargs: Dict[str, Any]) -> MutableMapping[str, Any]:
    """
    Inspect the input values received at the wrapper for the actual function call.

    :param param_names: parameter (*i.e.* argument) names of the original (decorated) function
    :param kwdefaults: default argument values of the original function
    :param args: arguments supplied to the call
    :param kwargs: keyword arguments supplied to the call
    :return: resolved arguments as they would be passed to the function
    """
    # pylint: disable=too-many-arguments
    mapping = dict()  # type: MutableMapping[str, Any]

    # Set the default argument values as condition parameters.
    for param_name, param_value in kwdefaults.items():
        mapping[param_name] = param_value

    # Override the defaults with the values actually suplied to the function.
    for i, func_arg in enumerate(args):
        mapping[param_names[i]] = func_arg

    for key, val in kwargs.items():
        mapping[key] = val

    return mapping
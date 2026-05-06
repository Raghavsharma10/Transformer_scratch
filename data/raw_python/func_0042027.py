def resolve_composed_functions(data, recursive=True):
    """
    Calls `ComposedFunction`s and returns its return value. By default, this
    function will recursively iterate dicts, lists, tuples, and sets and
    replace all `ComposedFunction`s with their return value.
    """

    if isinstance(data, ComposedFunction):
        data = data()

    if recursive:
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = resolve_composed_functions(
                    value,
                    recursive=recursive,
                )
        elif isinstance(data, (list, tuple, set)):
            for index, value in enumerate(data):
                data[index] = resolve_composed_functions(
                    value,
                    recursive=recursive,
                )

    return data
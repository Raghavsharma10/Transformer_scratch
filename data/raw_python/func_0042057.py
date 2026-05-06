def groupby(data, keys=None, size=None, min_size=None, max_size=None, contiguous=False):
    """
    :param data:
    :param keys:
    :param size:
    :param min_size:
    :param max_size:
    :param contiguous: MAINTAIN THE ORDER OF THE DATA, STARTING THE NEW GROUP WHEN THE SELECTOR CHANGES
    :return: return list of (keys, values) PAIRS, WHERE
                 keys IS IN LEAF FORM (FOR USE WITH {"eq": terms} OPERATOR
                 values IS GENERATOR OF ALL VALUE THAT MATCH keys
        contiguous -
    """
    if isinstance(data, Container):
        return data.groupby(keys)

    if size != None or min_size != None or max_size != None:
        if size != None:
            max_size = size
        return groupby_min_max_size(data, min_size=min_size, max_size=max_size)

    try:
        keys = listwrap(keys)
        if not contiguous:
            from jx_python import jx
            data = jx.sort(data, keys)

        if not data:
            return Null

        if any(is_expression(k) for k in keys):
            Log.error("can not handle expressions")
        else:
            accessor = jx_expression_to_function(jx_expression({"tuple": keys}))  # CAN RETURN Null, WHICH DOES NOT PLAY WELL WITH __cmp__

        def _output():
            start = 0
            prev = accessor(data[0])
            for i, d in enumerate(data):
                curr = accessor(d)
                if curr != prev:
                    group = {}
                    for k, gg in zip(keys, prev):
                        group[k] = gg
                    yield Data(group), data[start:i:]
                    start = i
                    prev = curr
            group = {}
            for k, gg in zip(keys, prev):
                group[k] = gg
            yield Data(group), data[start::]

        return _output()
    except Exception as e:
        Log.error("Problem grouping", cause=e)
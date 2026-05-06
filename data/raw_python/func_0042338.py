def _tuple_deep(v, field, depth, record):
    """
    field = {"name":name, "value":["attribute", "path"]}
    r[field.name]=v[field.value], BUT WE MUST DEAL WITH POSSIBLE LIST IN field.value PATH
    """
    if hasattr(field.value, "__call__"):
        return 0, None, record + (field.value(v),)

    for i, f in enumerate(field.value[depth : len(field.value) - 1 :]):
        v = v.get(f)
        if is_list(v):
            return depth + i + 1, v, record

    f = field.value.last()
    return 0, None, record + (v.get(f),)
def _select_deep(v, field, depth, record):
    """
    field = {"name":name, "value":["attribute", "path"]}
    r[field.name]=v[field.value], BUT WE MUST DEAL WITH POSSIBLE LIST IN field.value PATH
    """
    if hasattr(field.value, "__call__"):
        try:
            record[field.name] = field.value(wrap(v))
        except Exception as e:
            record[field.name] = None
        return 0, None

    for i, f in enumerate(field.value[depth : len(field.value) - 1 :]):
        v = v.get(f)
        if v is None:
            return 0, None
        if is_list(v):
            return depth + i + 1, v

    f = field.value.last()
    try:
        if not f:  # NO NAME FIELD INDICATES SELECT VALUE
            record[field.name] = v
        else:
            record[field.name] = v.get(f)
    except Exception as e:
        Log.error(
            "{{value}} does not have {{field}} property", value=v, field=f, cause=e
        )
    return 0, None
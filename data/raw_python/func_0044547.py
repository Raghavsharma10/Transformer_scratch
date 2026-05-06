def fmap(func, obj):
    """Creates a copy of obj with func applied to its contents."""
    if _coconut.hasattr(obj, "__fmap__"):
        return obj.__fmap__(func)
    args = _coconut_map(func, obj)
    if _coconut.isinstance(obj, _coconut.dict):
        args = _coconut_zip(args, obj.values())
    if _coconut.isinstance(obj, _coconut.tuple) and _coconut.hasattr(obj, "_make"):
        return obj._make(args)
    if _coconut.isinstance(obj, (_coconut.map, _coconut.range)):
        return args
    if _coconut.isinstance(obj, _coconut.str):
        return "".join(args)
    return obj.__class__(args)
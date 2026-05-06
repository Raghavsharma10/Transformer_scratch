def referents(igt, id, refattrs=None):
    """
    Return a list of ids denoting objects (tiers or items) in `igt` that
    are referred by the object denoted by `id` using a reference
    attribute in `refattrs`. If `refattrs` is None, then consider all
    known reference attributes for the type of object denoted by _id_.
    In other words, if 'b1' refers to 'a1' using 'alignment', then
    `referents(igt, 'b1', ['alignment'])` returns `['a1']`.
    """
    obj = igt.get_any(id)
    if obj is None:
        raise XigtLookupError(id)
    if refattrs is None:
        refattrs = obj.allowed_reference_attributes()
    return {ra: ids(obj.attributes.get(ra, '')) for ra in refattrs}
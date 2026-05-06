def dump_key(engine, obj):
    """dump the hash (and range, if there is one) key(s) of an object into
    a dynamo-friendly format.

    returns {dynamo_name: {type: value} for dynamo_name in hash/range keys}
    """
    key = {}
    for key_column in obj.Meta.keys:
        key_value = getattr(obj, key_column.name, missing)
        if key_value is missing:
            raise MissingKey("{!r} is missing {}: {!r}".format(
                obj, "hash_key" if key_column.hash_key else "range_key",
                key_column.name
            ))
        # noinspection PyProtectedMember
        key_value = engine._dump(key_column.typedef, key_value)
        key[key_column.dynamo_name] = key_value
    return key